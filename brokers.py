# ============================================================
#  brokers.py — Multi-Broker abstraction layer
# ============================================================
import logging
import time
import traceback
from abc import ABC, abstractmethod
import config

log = logging.getLogger(__name__)

class BaseBroker(ABC):
    """Common interface for all brokers."""
    
    @abstractmethod
    def place_order(self, symbol, quantity, side, product="MIS", order_type="MARKET",
                    strike=None, expiry=None, lot_size=None):
        """Place an order and return the order ID."""
        pass

    @abstractmethod
    def get_positions(self):
        """Return current net positions."""
        pass

    @abstractmethod
    def get_ltp(self, symbol):
        """Return Last Traded Price for a symbol."""
        pass
        
    @abstractmethod
    def get_balance(self):
        """Return available cash balance."""
        pass


class KiteBroker(BaseBroker):
    """Zerodha Kite implementation."""
    
    def __init__(self, kite_instance):
        self.kite = kite_instance
        self.name = "Zerodha Kite"

    def place_order(self, symbol, quantity, side, product="MIS", order_type="MARKET",
                    strike=None, expiry=None, lot_size=None):
        try:
            # side: 'BUY' or 'SELL'
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
            product_type = self.kite.PRODUCT_MIS if product == "MIS" else self.kite.PRODUCT_NRML
            ot = self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT

            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NFO,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product_type,
                order_type=ot
            )
            log.info(f"[{self.name}] Order placed: {side} {symbol} x {quantity} | ID: {order_id}")
            return order_id
        except Exception as e:
            log.error(f"[{self.name}] Order failed: {e}")
            return None

    def get_positions(self):
        try:
            return self.kite.positions()
        except Exception as e:
            log.error(f"[{self.name}] Failed to get positions: {e}")
            return {}

    def get_ltp(self, symbol):
        try:
            full_symbol = f"NFO:{symbol}"
            quote = self.kite.quote(full_symbol)
            return quote[full_symbol]["last_price"]
        except Exception as e:
            log.error(f"[{self.name}] Failed to get LTP for {symbol}: {e}")
            return 0.0

    def get_balance(self):
        try:
            margins = self.kite.margins()
            return float(margins["equity"]["available"]["cash"])
        except Exception as e:
            log.error(f"[{self.name}] Failed to get balance: {e}")
            return 0.0


class NeoBroker(BaseBroker):
    """Kotak Neo implementation."""
    
    def __init__(self, neo_client):
        self.client = neo_client
        self.name = "Kotak Neo"
        self._symbol_cache = {}  # Cache: {zerodha_symbol: kotak_symbol}

    def place_order(self, symbol, quantity, side, product="MIS", order_type="MARKET",
                    strike=None, expiry=None, lot_size=None):
        import kotak_auth
        import data_collector as dc

        # ── Kotak Neo NFO: SDK expects quantity as a STRING of total contracts ──
        # Kotak server error 1009 = quantity is not a valid lot-wise multiple.
        # quantity passed in = lots × lot_size (e.g. 1 lot of Nifty = 1 × 75 = 75)
        # lot_size passed in = position.lot_size (sourced from config.LOT_SIZE = 75)
        #
        # IMPORTANT: Never fall back to hardcoded 25 — Nifty lot size is now 75.
        # Always use the lot_size argument; fall back to config only if arg is zero/None.
        effective_lot_size = int(lot_size or config.LOT_SIZE)
        if effective_lot_size <= 0:
            effective_lot_size = config.LOT_SIZE
        # Derive number of lots; clamp to at least 1
        num_lots = max(1, int(quantity) // effective_lot_size)
        # Total contracts sent to Kotak must be an exact lot multiple, as a string
        neo_qty = str(num_lots * effective_lot_size)
        log.info(
            f"[{self.name}] Qty calc: {quantity} contracts ÷ lot_size({effective_lot_size}) "
            f"= {num_lots} lot(s) → neo_qty='{neo_qty}' (string sent to Kotak)"
        )

        # ── 1. Symbol Translation (Fast Path) ────────────────────
        # Skip JIT search if we already have a mapping in cache.
        if symbol in self._symbol_cache:
            log.debug(f"[{self.name}] Symbol Cache Hit: {symbol} ➔ {self._symbol_cache[symbol]}")
            symbol = self._symbol_cache[symbol]
        else:
            # Fallback to JIT search (slow) if not cached (e.g. for re-entries)
            symbol = self._get_mapped_symbol(symbol, strike, expiry)

        for attempt in range(2):  # Try twice: once normally, once after session refresh
            try:
                # Parameter mapping for Neo SDK (values confirmed from API error messages)
                t_side = 'B' if side == "BUY" else 'S'
                p_type = 'MIS' if product == "MIS" else 'NRML'
                o_type = 'MKT' if order_type == "MARKET" else 'L'

                # ── Diagnostic Logs (User Requested) ──
                # Access token and host from SDK internal configuration
                cur_token = self.client.configuration.edit_token or ""
                cur_host = getattr(self.client.configuration, "host", "") or getattr(self.client.api_client.configuration, "host", "")
                print(f"   ORDER TOKEN: {cur_token[:20]}...")
                print(f"   ORDER BASE URL: {cur_host}")
                log.info(f"Order Diagnostics: host={cur_host} token={cur_token[:15]}...")

                payload = {
                    "exchange_segment": 'nfo', "product": p_type, "price": '0',
                    "order_type": o_type, "quantity": neo_qty, "validity": 'DAY',
                    "trading_symbol": symbol, "transaction_type": t_side, "amo": 'NO'
                }
                log.info(f"[{self.name}] Attempt {attempt+1}: Sending payload: {payload}")

                response = self.client.place_order(**payload)
                log.info(f"[{self.name}] RAW API RESPONSE: {response}")

                # Detect unauthorized / session-expired response
                resp_str = str(response).strip().lower()
                is_unauth = (
                    resp_str == "unauthorized"
                    or (isinstance(response, dict) and
                        str(response.get("message", "") or response.get("errMsg", "")).lower() == "unauthorized")
                )

                if is_unauth:
                    if attempt == 0:
                        log.warning(f"[{self.name}] Unauthorized on attempt 1 — refreshing session...")
                        print(f"  🔄 [{self.name}] Session expired. Auto re-logging in...")
                        # ── Force a FRESH client from the auth module ──
                        fresh_client = kotak_auth.get_neo_client(force_relogin=True)
                        if fresh_client:
                            self.client = fresh_client # Update the broker's client reference
                            log.info(f"[{self.name}] Re-login successful. Waiting 3s for session sync...")
                            time.sleep(3) # Support suggested 3-5 sec delay
                            continue
                        else:
                            log.error(f"[{self.name}] Re-login failed during recovery.")
                            return None
                    else:
                        log.error(f"[{self.name}] Still unauthorized after re-login. Aborting.")
                        return None

                # Parse order ID from response
                order_id = None
                if isinstance(response, dict):
                    order_id = (response.get("orderId") or response.get("nOrdNo")
                                or response.get("order_id"))
                    if not order_id:
                        err_msg = response.get("message") or response.get("errMsg") or str(response)
                        log.error(f"[{self.name}] Order response missing ID: {response}")
                        print(f"  ❌ [{self.name}] REJECTED: {err_msg}")
                        return None
                else:
                    # Raw string response — if not an error, treat as order ID
                    if any(e in resp_str for e in ["error", "invalid", "fail", "not logged"]):
                        log.error(f"[{self.name}] Order error string: {response}")
                        print(f"  ❌ [{self.name}] REJECTED: {response}")
                        return None
                    order_id = response

                log.info(f"[{self.name}] Order placed: {side} {symbol} x {quantity} | ID: {order_id}")
                return order_id

            except Exception as e:
                log.error(f"[{self.name}] API Exception (at={attempt+1}): {e}\n{traceback.format_exc()}")
                if attempt == 0:
                    time.sleep(1)
        return None

    def get_positions(self):
        try:
            # Using Neo SDK's positions method
            return self.client.get_positions()
        except Exception as e:
            log.error(f"[{self.name}] Failed to get positions: {e}")
            return []

    def get_ltp(self, symbol):
        try:
            # v2: Get quotes using instrument search
            # Try NFO first (Derivatives), fallback to NSE_CM (Spot/Equity)
            segments = ['nfo', 'nse_cm']
            res_list = []
            final_segment = 'nfo'
            
            for seg in segments:
                scrip = self.client.search_scrip(exchange_segment=seg, symbol=symbol)
                res_list = scrip.get("data", []) if isinstance(scrip, dict) else scrip
                if isinstance(res_list, list) and len(res_list) > 0:
                    final_segment = seg
                    break
            
            if isinstance(res_list, list) and len(res_list) > 0:
                # 🧪  Use 'tk' or 'pToken' depending on NAPI version
                token = res_list[0].get("tk") or res_list[0].get("pToken")
                
                req_tokens = [{'instrument_token': str(token), 'exchange_segment': final_segment}]
                res = self.client.quotes(instrument_tokens=req_tokens, quote_type='ltp')
                
                # Check 'data' first, then 'items' (SDK inconsistency)
                q_list = []
                if isinstance(res, dict):
                    q_list = res.get("data", []) or res.get("items", [])
                elif isinstance(res, list):
                    q_list = res
                
                if q_list and len(q_list) > 0:
                    # 'lp' or 'ltp' or 'lval' for NAPI LTP
                    lp = q_list[0].get("lp") or q_list[0].get("ltp") or q_list[0].get("lval")
                    if lp is not None:
                        return float(lp)
            return 0.0
        except Exception as e:
            log.error(f"[{self.name}] Failed to get LTP for {symbol}: {e}")
            return 0.0

    def get_balance(self):
        try:
            # v2: limits() returns account balance information
            res = self.client.limits()
            if isinstance(res, dict) and "data" in res:
                # 'availableMargin' is a common field in Neo response
                return float(res["data"].get("availableMargin", 0.0))
            return 0.0
        except Exception as e:
            log.error(f"[{self.name}] Failed to get balance: {e}")
            return 0.0

    # ── LATENCY OPTIMIZATION: Symbol Caching Mechanism ──────────
    
    def _get_mapped_symbol(self, symbol, strike, expiry) -> str:
        """Translates a Zerodha symbol to Kotak Neo using search_scrip (slow). Stores in cache."""
        try:
            # ── FAST-PATH: Bypass API for standard NFO symbols ───────────
            # If the symbol already looks like a valid exchange trading symbol 
            # (e.g. 'NIFTY2640722200CE'), we assume it matches Kotak's backend 
            # directly and skip the slow search_scrip call.
            s_clean = str(symbol or "").strip()
            if s_clean.startswith("NIFTY") and " " not in s_clean and len(s_clean) >= 14:
                log.info(f"[{self.name}] [FAST-PATH] Using standard symbol: {s_clean}")
                self._symbol_cache[symbol] = s_clean
                return s_clean

            # ── SLOW-PATH: Fallback to JIT Search ───────────────────────
            search_query = f"NIFTY {int(float(strike or 0))}"
            log.info(f"[{self.name}] [LATENCY] Performing JIT Search for '{search_query}'...")
            
            scrip_box = self.client.search_scrip(exchange_segment='nfo', symbol=search_query)
            scrip_data = scrip_box.get("data", []) if isinstance(scrip_box, dict) else scrip_box
            
            if scrip_data and isinstance(scrip_data, list):
                target_type = 'CE' if str(symbol).strip().endswith('CE') else 'PE'
                exp_needle = expiry.strftime("%d%b%y").upper() if expiry else ""
                
                for candidate in scrip_data:
                    c_sym = str(candidate.get("pTrdSymbol") or candidate.get("stk") or "").strip()
                    if target_type in c_sym and exp_needle in c_sym.upper():
                        k_symbol = candidate.get("pTrdSymbol") or candidate.get("stk")
                        log.info(f"[{self.name}] Cache Map: {symbol} ➔ {k_symbol}")
                        self._symbol_cache[symbol] = k_symbol
                        return k_symbol

            log.warning(f"[{self.name}] No match for {symbol} JIT search. Caching original as fallback.")
            self._symbol_cache[symbol] = symbol
            return symbol
        except Exception as e:
            log.error(f"[{self.name}] Symbol mapping failed: {e}")
            return symbol

    def pre_map_symbols(self, strikes_list, expiry_date):
        """Pre-warms the symbol cache for all candidate strikes (call at 9:15 AM)."""
        if not strikes_list: return
        log.info(f"[{self.name}] 🚀 Pre-warming symbol cache for {len(strikes_list)} strikes...")
        start_time = time.time()
        
        for s in strikes_list:
            strike_val = s['strike']
            # CE
            self._get_mapped_symbol(s['ce_symbol'], strike_val, expiry_date)
            # PE
            self._get_mapped_symbol(s['pe_symbol'], strike_val, expiry_date)
            
        elapsed = time.time() - start_time
        log.info(f"[{self.name}] ✅ Symbol cache pre-warmed in {elapsed:.1f}s. {len(self._symbol_cache)} mappings stored.")