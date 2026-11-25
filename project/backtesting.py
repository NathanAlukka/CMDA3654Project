import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_backtest(
    df,
    entry_threshold=1.5,
    exit_threshold=0.5,
    test_start="2023-01-01",
    test_end="2026-11-01",
    initial_capital=100000
):
    """
    Backtests the pairs trading strategy using the z-scores
    created by the main pairs-trading model.

    position_y:
        1  = long V, short MA
        -1 = short V, long MA
        0  = no position
    """

    # Only evaluate the strategy on the test period
    bt = df.loc[test_start:test_end].copy()

    # The main notebook stores MA and V as log prices.
    # Convert them back to actual prices for return calculations.
    bt["MA_price"] = np.exp(bt["MA"])
    bt["V_price"] = np.exp(bt["V"])

    # ---------------------------------------------------------
    # 1. BUILD TRADING POSITIONS
    # ---------------------------------------------------------

    positions_y = []
    current_position = 0

    for _, row in bt.iterrows():

        z = row["zscore"]
        cointegrated = row["cointegrated"]

        # If pair is no longer cointegrated, close the trade
        if pd.isna(z) or cointegrated != 1:
            current_position = 0

        # No current trade: check entry conditions
        elif current_position == 0:

            # V relatively underpriced
            if z < -entry_threshold:
                current_position = 1

            # V relatively overpriced
            elif z > entry_threshold:
                current_position = -1

        # Currently long V / short MA
        elif current_position == 1:

            # Spread has reverted toward normal
            if z > -exit_threshold:
                current_position = 0

        # Currently short V / long MA
        elif current_position == -1:

            # Spread has reverted toward normal
            if z < exit_threshold:
                current_position = 0

        positions_y.append(current_position)

    bt["bt_position_y"] = positions_y
    bt["bt_position_x"] = -bt["bt_position_y"]

    # ---------------------------------------------------------
    # 2. CALCULATE STOCK RETURNS
    # ---------------------------------------------------------

    bt["MA_return"] = bt["MA_price"].pct_change()
    bt["V_return"] = bt["V_price"].pct_change()

    # Shift positions by one day.
    # Today's signal is used for the NEXT day's return.
    # This prevents look-ahead bias.
    bt["effective_position_x"] = bt["bt_position_x"].shift(1).fillna(0)
    bt["effective_position_y"] = bt["bt_position_y"].shift(1).fillna(0)

    # Equal allocation to both sides of the pair
    bt["strategy_return"] = (
        0.5 * bt["effective_position_x"] * bt["MA_return"]
        + 0.5 * bt["effective_position_y"] * bt["V_return"]
    )

    # ---------------------------------------------------------
    # 3. PORTFOLIO VALUE
    # ---------------------------------------------------------

    bt["portfolio_value"] = (
        initial_capital
        * (1 + bt["strategy_return"]).cumprod()
    )

    # ---------------------------------------------------------
    # 4. DRAWDOWN
    # ---------------------------------------------------------

    bt["running_max"] = bt["portfolio_value"].cummax()

    bt["drawdown"] = (
        bt["portfolio_value"] / bt["running_max"] - 1
    )

    # ---------------------------------------------------------
    # 5. IDENTIFY INDIVIDUAL TRADES
    # ---------------------------------------------------------

    active_position = bt["effective_position_y"]

    new_trade = (
        (active_position != 0)
        & (active_position != active_position.shift(1))
    )

    bt["trade_id"] = new_trade.cumsum()

    trade_data = bt[active_position != 0]

    trade_returns = (
        trade_data.groupby("trade_id")["strategy_return"]
        .apply(lambda x: (1 + x).prod() - 1)
    )

    # ---------------------------------------------------------
    # 6. PERFORMANCE METRICS
    # ---------------------------------------------------------

    total_return = (
        bt["portfolio_value"].iloc[-1] / initial_capital - 1
    )

    number_of_trades = len(trade_returns)

    winning_trades = (trade_returns > 0).sum()
    losing_trades = (trade_returns < 0).sum()

    if number_of_trades > 0:
        win_rate = winning_trades / number_of_trades
        average_trade_return = trade_returns.mean()
    else:
        win_rate = np.nan
        average_trade_return = np.nan

    daily_returns = bt["strategy_return"].dropna()

    if daily_returns.std() != 0:
        sharpe_ratio = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )
    else:
        sharpe_ratio = np.nan

    annualized_volatility = (
        daily_returns.std() * np.sqrt(252)
    )

    max_drawdown = bt["drawdown"].min()

    # ---------------------------------------------------------
    # 7. PERFORMANCE SUMMARY
    # ---------------------------------------------------------

    summary = pd.DataFrame({
        "Metric": [
            "Initial Portfolio",
            "Final Portfolio",
            "Total Return",
            "Number of Trades",
            "Winning Trades",
            "Losing Trades",
            "Win Rate",
            "Average Trade Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown"
        ],

        "Value": [
            initial_capital,
            bt["portfolio_value"].iloc[-1],
            total_return,
            number_of_trades,
            winning_trades,
            losing_trades,
            win_rate,
            average_trade_return,
            annualized_volatility,
            sharpe_ratio,
            max_drawdown
        ]
    })

    print("\nPAIRS TRADING BACKTEST RESULTS")
    print("--------------------------------")

    print(f"Initial Portfolio:     ${initial_capital:,.2f}")
    print(f"Final Portfolio:       ${bt['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total Return:          {total_return:.2%}")
    print(f"Number of Trades:      {number_of_trades}")
    print(f"Winning Trades:        {winning_trades}")
    print(f"Losing Trades:         {losing_trades}")
    print(f"Win Rate:              {win_rate:.2%}")
    print(f"Average Trade Return:  {average_trade_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio:          {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown:      {max_drawdown:.2%}")

    # ---------------------------------------------------------
    # 8. PERFORMANCE GRAPHS
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        bt.index,
        bt["portfolio_value"],
        label="Pairs Trading Portfolio"
    )

    plt.axhline(
        initial_capital,
        linestyle="--",
        label="Initial Capital"
    )

    plt.title("Pairs Trading Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 5))

    plt.plot(
        bt.index,
        bt["drawdown"]
    )

    plt.title("Pairs Trading Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.show()

    return bt, summary, trade_returns