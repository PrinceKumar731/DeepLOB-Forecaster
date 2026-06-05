import numpy as np
import pandas as pd

class LOBBacktester:
    """
    Backtests order book trading strategies under execution latency, transaction fees, and slippage.
    """
    def __init__(self, initial_capital=100000.0, latency_ms=100, transaction_fee=0.0005, slippage=0.0001, tick_interval_ms=100):
        self.initial_capital = initial_capital
        # Latency is converted to steps/ticks. E.g. 100ms latency on a 100ms stream = 1 step delay.
        self.latency_steps = max(1, int(latency_ms / tick_interval_ms))
        self.transaction_fee = transaction_fee
        self.slippage = slippage

    def run(self, df, predictions):
        """
        Runs the simulation.
        df: pandas DataFrame containing LOB states. Must have 'timestamp', 'bid_p_0', 'ask_p_0', 'mid_price'.
        predictions: numpy array of predictions [Down=0, Flat=1, Up=2] corresponding to each row of df.
        
        Trading strategy:
        - Signal Up (2): Long position (+1)
        - Signal Down (0): Short position (-1)
        - Signal Flat (1): No position (0)
        """
        N = len(df)
        if len(predictions) != N:
            raise ValueError(f"Length mismatch: df ({N}) vs predictions ({len(predictions)})")

        timestamps = df['timestamp'].values
        bid_prices = df['bid_p_0'].values
        ask_prices = df['ask_p_0'].values
        mid_prices = df['mid_price'].values

        capital = self.initial_capital
        position = 0  # 0: Flat, 1: Long, -1: Short
        position_qty = 0.0
        entry_price = 0.0
        
        equity_curve = np.zeros(N)
        positions = np.zeros(N)
        trades = []  # Logs all trades

        # Cache values to speed up loop
        for t in range(N):
            # Calculate current portfolio value (mark-to-market)
            mid = mid_prices[t]
            if position == 1:
                # Value if we liquidated at best bid (taking fees and slippage into account)
                liquid_price = bid_prices[t] * (1 - self.slippage)
                portfolio_value = capital + position_qty * (liquid_price - entry_price)
            elif position == -1:
                # Value if we cover at best ask
                liquid_price = ask_prices[t] * (1 + self.slippage)
                portfolio_value = capital + position_qty * (entry_price - liquid_price)
            else:
                portfolio_value = capital
                
            equity_curve[t] = portfolio_value
            positions[t] = position

            # Signal generation at time t (only if we have room for latency)
            if t + self.latency_steps >= N:
                continue

            target_pos = 0
            signal = predictions[t]
            if signal == 2:
                target_pos = 1
            elif signal == 0:
                target_pos = -1
            elif signal == 1:
                target_pos = 0

            # Execute order at t + latency
            exec_t = t + self.latency_steps
            exec_bid = bid_prices[exec_t]
            exec_ask = ask_prices[exec_t]
            exec_time = timestamps[exec_t]

            # If position change is required
            if target_pos != position:
                # Liquidate old position
                if position != 0:
                    exit_price = exec_bid * (1 - self.slippage) if position == 1 else exec_ask * (1 + self.slippage)
                    pnl = position_qty * (exit_price - entry_price) if position == 1 else position_qty * (entry_price - exit_price)
                    fee = exit_price * position_qty * self.transaction_fee
                    capital += pnl - fee
                    
                    trades.append({
                        "type": "Exit",
                        "direction": "Long" if position == 1 else "Short",
                        "entry_time": timestamps[t],
                        "exit_time": exec_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": position_qty,
                        "pnl": pnl - fee,
                        "fee": fee,
                        "capital_after": capital
                    })
                    
                    position = 0
                    position_qty = 0.0

                # Enter new position
                if target_pos != 0:
                    position = target_pos
                    # Buy at ask (if long) or sell at bid (if short)
                    entry_raw_price = exec_ask if position == 1 else exec_bid
                    entry_price = entry_raw_price * (1 + self.slippage) if position == 1 else entry_raw_price * (1 - self.slippage)
                    
                    # Size the position: allocate 95% of capital to account for fees/slippage buffer
                    allocated_capital = capital * 0.95
                    position_qty = allocated_capital / entry_price
                    fee = entry_price * position_qty * self.transaction_fee
                    capital -= fee  # Pay entry fee immediately

        # Compile final stats
        trades_df = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve)
        
        # Performance metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Daily Sharpe calculation (mocked for high frequency returns)
        # We calculate tick-to-tick return standard deviation and scale
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            # Scale by sqrt(number of ticks in a year)
            # Assuming 100ms intervals, there are 10 * 60 * 60 * 24 * 365 = 315,360,000 ticks in a year
            # Let's use a standard scaling factor for a active trading session: sqrt(252 * 8 * 60 * 60 * 10) for 100ms ticks
            annualized_factor = np.sqrt(252 * 24 * 60 * 60 * 10)  # 252 days, 24 hours, 3600 seconds * 10 ticks/s
            sharpe = (returns.mean() / returns.std()) * annualized_factor
        else:
            sharpe = 0.0

        # Drawdown calculation
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        win_rate = 0.0
        if not trades_df.empty:
            profitable_trades = (trades_df['pnl'] > 0).sum()
            win_rate = profitable_trades / len(trades_df)

        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": equity_curve[-1],
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": len(trades_df),
            "win_rate_pct": win_rate * 100
        }

        return metrics, equity_series, trades_df
