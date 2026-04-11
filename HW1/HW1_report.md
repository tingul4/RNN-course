# Assignment 1: Stock Prediction & Live Trading Competition

**Name:** 陳廷曜  
**Student ID:** 314832008  
**Target Stock:** TSMC (2330.TW)  
**Source Code:** [Stock_predict_final.ipynb](Stock_predict_final.ipynb), ablation study in [ablation_exp.ipynb](ablation_exp.ipynb)

---

## 1. Model Analysis

### 1.1 Data Pipeline and Final Model

I collected about 10 years of daily data for `2330.TW` using `yfinance`.

The three features, `Close`, `Volume`, and `MA5`, were each normalized to `[0, 1]` using separate `MinMaxScaler`s. This turned out to be the most important preprocessing decision. Since raw `Volume` is on the order of 20 million while `Close` is around 1,800, using a shared scaler would allow `Volume` to dominate the dynamic range and make the model largely ignore the price information.

The final Attention-LSTM architecture is shown below:

```text
Input (look_back=90, 3 features)
  → LSTM(128, return_sequences=True)
  → LSTM(64,  return_sequences=True)
  → Attention  (custom layer: tanh → softmax over time)
  → Dense(1)
```

- **Optimizer:** Adam, lr = 1e-3
- **Loss:** MSE
- **Epochs / Batch size:** 50 / 32
- **Train/test split:** the last 5% of the time series as the test set (`train_test_split(..., shuffle=False)`)

Why add Attention on top of stacked LSTMs? Once I extended `look_back` to 90 days, the earlier observations in each window tended to be compressed or forgotten by a vanilla LSTM. The attention layer allows the model to reweight all 90 time steps and place more emphasis on the days that matter most, such as breakout candles or high-volume sessions, instead of relying only on the final hidden state.

### 1.2 Baseline vs. Best Model

For the baseline, I used a minimal setup: only `Close`, `look_back=60`, no dropout, and all other settings identical to the final model. This represents a sequence model with no additional feature engineering and no tuning of the window size.

| Model Configuration | Features | `look_back` | Dropout | Test RMSE | Test MAPE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline (Close only) | `[Close]` | 60 | 0.0 | 46.09 | 2.23% |
| **Best Model** | `[Close, Volume, MA5]` | **90** | **0.0** | **35.14** | **1.74%** |

Compared with the baseline, the best model reduces RMSE by about **23.8%** and MAPE by about **22.0%**. The next section supports this conclusion with an ablation study rather than relying only on a single best result.

### 1.3 Ablation Study

I ran a full grid search over three factors, resulting in \(4 \times 3 \times 2 = 24\) experiments:

- **Features:** `[Close]`, `[Close, Volume]`, `[Close, MA5]`, `[Close, Volume, MA5]`
- **look_back:** 60 / 90 / 120
- **Dropout:** 0.0 / 0.2

Every run used 50 epochs, batch size 32, and `seed=2026` for reproducibility. The full results are available in [ablation_exp.ipynb](ablation_exp.ipynb), sorted by MAPE:

| Rank | Features | LB | Dropout | RMSE | MAPE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Close, Vol, MA5** | **90** | **0.0** | **35.14** | **1.74** |
| 2 | Close | 90 | 0.0 | 37.14 | 1.81 |
| 3 | Close, Vol | 90 | 0.0 | 36.99 | 1.82 |
| 4 | Close, Vol | 120 | 0.0 | 41.38 | 1.98 |
| 5 | Close, MA5 | 90 | 0.0 | 44.66 | 2.11 |
| 6 | Close, MA5 | 120 | 0.0 | 47.29 | 2.20 |
| 7 | Close | 120 | 0.0 | 47.78 | 2.23 |
| 8 | Close | 60 | 0.0 | 46.09 | 2.23 |
| 9 | Close | 90 | 0.2 | 55.08 | 2.65 |
| 10 | Close, Vol, MA5 | 60 | 0.0 | 57.91 | 2.81 |
| 11 | Close, MA5 | 60 | 0.0 | 55.74 | 2.82 |
| 12 | Close, Vol, MA5 | 120 | 0.0 | 61.23 | 3.00 |
| 13 | Close, Vol | 60 | 0.0 | 60.53 | 3.15 |
| 14 | Close, Vol, MA5 | 90 | 0.2 | 88.65 | 4.45 |
| 15 | Close, MA5 | 60 | 0.2 | 97.98 | 5.10 |
| 16 | Close, Vol | 120 | 0.2 | 105.42 | 5.25 |
| 17 | Close, Vol, MA5 | 60 | 0.2 | 98.15 | 5.28 |
| 18 | Close | 120 | 0.2 | 105.90 | 5.58 |
| 19 | Close, MA5 | 90 | 0.2 | 120.87 | 5.75 |
| 20 | Close | 60 | 0.2 | 108.67 | 5.90 |
| 21 | Close, Vol | 90 | 0.2 | 120.90 | 6.43 |
| 22 | Close, Vol | 60 | 0.2 | 122.73 | 6.85 |
| 23 | Close, Vol, MA5 | 120 | 0.2 | 138.42 | 6.98 |
| 24 | Close, MA5 | 120 | 0.2 | 150.63 | 8.12 |

**Why I chose these hyperparameters:**

1. **`look_back=90`.** Every one of the top 6 configurations uses either LB=90 or LB=120, and every LB=60 configuration has MAPE of at least 2.23%. Intuitively, 90 trading days cover roughly half a financial quarter plus several short-term swings. This is long enough to capture mid-term momentum, but not so long that the model starts relying on stale market behavior. Extending the window to 120 days slightly hurts performance, which suggests that the oldest 30 days contribute more noise than useful signal.

2. **`Close + Volume + MA5`.** In the dropout-zero, LB=90 setting, adding `Volume` and `MA5` improves RMSE from 37.14 to 35.14 and MAPE from 1.81% to 1.74%. The gain may look small, but it is averaged over a 117-day test period. In practice, that means the 3-feature model performs slightly better on many days, rather than winning because of one or two lucky predictions.

3. **No dropout.** All 12 runs with `dropout=0.2` fall between 2.65% and 8.12% MAPE, and every one performs worse than its `dropout=0.0` counterpart. This was the most surprising result of the assignment. I initially expected dropout to help because 90-day windows with stacked LSTMs seemed prone to overfitting. However, with only about 2300 rows, 50 epochs, and batch size 32, the model likely never enters a regime where overfitting becomes the main issue. In this case, dropout reduces model capacity before that capacity becomes harmful.

4. **Adding `Volume` alone helps more than adding `MA5` alone.** Rank 3 (`Close + Volume`, 1.82%) outperforms Rank 5 (`Close + MA5`, 2.11%). This makes sense because MA5 is a linear transformation of `Close` and therefore contains a large amount of redundant information. In contrast, `Volume` provides information that `Close` alone cannot capture, namely the strength or conviction behind a move. Still, using both together performs slightly better than using either one alone, because MA5 helps smooth short-term noise in a way that `Volume` does not.

### 1.4 Rolling Forecast Simulation

**Implementation.** In [Stock_predict_final.ipynb](Stock_predict_final.ipynb), I reserved the last 10 trading days before the training cutoff for rolling evaluation. On each day:

1. Refit the scalers on all data up to day \(t\), then extract the last `look_back=90` rows of `[Close, Volume, MA5]`.
2. Run a forward pass to obtain \(\hat{p}_{t+1}\), inverse-transform it, and record the prediction.
3. Append the true close of day \(t+1\), slide the window forward, and repeat.

**Result.** The rolling prediction curve tracks the actual closing price closely (see the `Rolling Prediction vs Actual` plot in the notebook), with daily errors remaining in a relatively small range. This gave me enough confidence to use the forecast as a directional signal in the live trading stage.

---

## 2. Trading Strategy & Log

### 2.1 Trading Strategy

My strategy can be summarized as **"let the model determine the direction, and let position sizing control the risk."** More concretely:

- **Signal.** After each market close, I run the rolling forecast pipeline to obtain \(\hat{p}_{t+1}\) and compute \(\Delta = (\hat{p}_{t+1} - p_t) / p_t\).
- **Direction.** If \(\Delta > +0.5\%\), I take a bullish view; if \(\Delta < -0.5\%\), I take a bearish view; otherwise, I hold.
- **Scale in gradually, not all at once.** Instead of deploying all available cash on the first bullish signal, I build the position in stages. This allows me to average into the position and prevents a single incorrect prediction from causing excessive damage.
- **Sell into strength, buy into weakness.** When the model projects a relatively high next-day price compared with my average cost, I take profit on part or all of the position. When the market goes through a meaningful pullback, I may also add shares at a lower executed price to improve the average cost basis.

The main idea is to convert the model's point forecast into a **range of conviction** rather than a binary trading trigger. A weak signal leads to a small position, while a strong signal justifies a larger adjustment in position size.

### 2.2 Trading Log

To avoid ambiguity, I separately report the executed price and the actual closing price. The executed price is the actual transaction price used to calculate cash flow, while the actual close is used to track market movement and mark the position to market.

| Date | Predicted | Executed Price | Actual Close | Action | Qty | Cash Flow (TWD) | Cash Balance (TWD) | Holdings |
| :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| **Initial** | – | – | – | – | 0 | 0 | 10,000,000 | 0 |
| **3/23** | 1821.37 | 1820.00 | 1810 | **Buy** | 1000 | −1,820,000 | 8,180,000 | 1000 |
| **3/24** | 1817.54 | 1815.00 | 1810 | **Buy** | 500 | −907,500 | 7,272,500 | 1500 |
| **3/25** | 1803.32 | 1805.00 | 1845 | **Buy** | 500 | −902,500 | 6,370,000 | 2000 |
| **3/26** | 1876.19 | 1875.00 | 1840 | **Sell** | 1000 | +1,875,000 | 8,245,000 | 1000 |
| **3/27** | 1841.65 | – | 1820 | **Hold** | 0 | 0 | 8,245,000 | 1000 |
| **3/30** | 1831.02 | – | 1780 | **Hold** | 0 | 0 | 8,245,000 | 1000 |
| **3/31** | 1836.48 | – | 1760 | **Hold** | 0 | 0 | 8,245,000 | 1000 |
| **4/01** | 1734.36 | 1735.00 | 1855 | **Buy** | 500 | −867,500 | 7,377,500 | 1500 |
| **4/02** | 1828.05 | 1830.00 | 1810 | **Sell** | 1500 | +2,745,000 | 10,122,500 | 0 |
| **4/07** | 1852.99 | – | 1860 | **Hold** | 0 | 0 | 10,122,500 | 0 |

The gradual scaling pattern is clear in this log: from 3/23 to 3/25, I built the position in three steps; on 3/26, I took partial profit after the rebound; from 3/27 to 3/31, I held the remaining position through the pullback; on 4/01, I added 500 shares; and on 4/02, I fully exited the position.

---

## 3. Reflection

### 3.1 Trading Performance

- **Final total asset:** 10,122,500 TWD
- **Return on investment (ROI):** **+1.225%** over 10 trading days
- **Max drawdown (MDD):** approximately **0.79%**. The worst unrealized loss occurred around **3/31**, when the remaining position experienced the deepest pullback during the holding period. Because each entry used only a modest portion of total capital, the drawdown remained limited and the portfolio still finished the period with a positive return.

### 3.2 Did the Model Work?

**Somewhere between "often" and "sometimes."** The model was more useful for identifying short-term directional bias than for predicting the exact closing price. In particular, it helped me build and reduce positions with more confidence during the main swings in the trading window.

For example, the predicted rebound around **3/26** supported my decision to take partial profit after the position had been built over the previous three sessions. More generally, the rolling forecast results suggested that the model could track short-term price movement reasonably well, even when the magnitude of the prediction was not perfectly accurate.

The relatively low error level in the rolling experiment was what convinced me that the model was reliable enough for real position sizing. Without that validation, I would probably have traded much more conservatively, perhaps only 200 to 300 shares per trade instead of 500 to 1000, and the final ROI would likely have been smaller.

### 3.3 Did I Follow the Model Strictly, or Intervene Manually?

**I followed the model's directional signal closely, but position sizing and final exit timing were still discretionary decisions.**

The clearest example of manual intervention was the **full liquidation on 4/02**. Even though the model continued to provide useful directional information, I still chose to lock in profits and fully exit the position once the trade had already produced a satisfactory result. This decision was based more on risk control than on blindly following the model output.

In my view, the model tells me *where the market is more likely to go*, but risk management determines *when I should override the model*.

### 3.4 Conclusion

Overall, the model performed better than I expected. Although its predictions still contain noticeable errors and are not reliable enough to be applied directly to real-world trading without caution, they nevertheless provide useful information about likely market direction.

Stock prices are influenced by too many factors to be predicted perfectly by a single model. Even so, this project showed that an LSTM-based approach can still provide meaningful value as a decision-support tool in short-term trading.