# ============================================================
#  brokers.py — Multi-Broker abstraction layer
# ============================================================
import logging
import time
from abc import ABC, abstractmethod
import config

log = logging.getLogger(__name__)

class BaseBroker(ABC):
    """Common interface for all brokers."""
    
    @abstractmethod
    def place_order(self, symbol, quantity, side, product="MIS", order_type="MARKET",
                    strike=None, expiry=None):
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
                    strike=None, expiry=None):
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

    def place_order(self, symbol, quantity, side, product="MIS", order_type="MARKET",
                    strike=None, expiry=None):
        import kotak_auth
        import data_collector as dc

        # ── Robust Just-in-Time (JIT) Symbol Translation ──────
        try:
            # Most reliable method: search for "NIFTY <STRIKE>"
            # e.g., "NIFTY 22800" will return CE/PE for various expiries.
            search_query = f"NIFTY {int(float(strike or 0))}"
            log.info(f"[{self.name}] JIT Search Query: '{search_query}'")
            
            scrip_box = self.client.search_scrip(exchange_segment='nfo', symbol=search_query)
            scrip_data = scrip_box.get("data", []) if isinstance(scrip_box, dict) else scrip_box
            
            if scrip_data and isinstance(scrip_data, list):
                # Filter results to find the best match for current expiry and type
                target_type = 'CE' if str(symbol).strip().endswith('CE') else 'PE'
                # Format expiry for matching: Kotak often uses 'DDMMMYY' (e.g. 07APR26)
                exp_needle = expiry.strftime("%d%b%y").upper() if expiry else ""
                
                match = None
                for candidate in scrip_data:
                    c_sym = str(candidate.get("pTrdSymbol") or candidate.get("stk") or "").strip()
                    # Check if candidate matches our Type and Expiry
                    if target_type in c_sym and exp_needle in c_sym.upper():
                        match = candidate
                        break
                
                if match:
                    k_symbol = match.get("pTrdSymbol") or match.get("stk")
                    log.info(f"[{self.name}] JIT Match Found: {symbol} ➔ {k_symbol} (Token: {match.get('pToken')})")
                    symbol = k_symbol
                else:
                    log.warning(f"[{self.name}] JIT: Found {len(scrip_data)} results for {search_query} but none matched expiry {exp_needle}. Proceeding with: {symbol}")
            else:
                log.warning(f"[{self.name}] JIT: search_scrip returned no data for {search_query}. Using original: {symbol}")
        except Exception as e:
            log.error(f"[{self.name}] JIT lookup failed unexpectedly: {e}")

        for attempt in range(2):  # Try twice: once normally, once after session refresh
            try:
                # Parameter mapping for Neo SDK (values confirmed from API error messages)
                t_side = 'B' if side == "BUY" else 'S'
                p_type = 'MIS' if product == "MIS" else 'NRML'
                o_type = 'MKT' if order_type == "MARKET" else 'L'

                payload = {
                    "exchange_segment": 'nfo', "product": p_type, "price": '0',
                    "order_type": o_type, "quantity": str(quantity), "validity": 'DAY',
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
                            log.info(f"[{self.name}] Re-login successful. Retrying order...")
                            time.sleep(2) # Give server a moment to sync session
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