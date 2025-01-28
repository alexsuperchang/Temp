"""
完整版量化交易系统(含价格建议与止损策略)
新增功能：动态止盈止损、波动率自适应价格建议、风险调整仓位管理
"""
# ...（保留之前导入的套件，新增以下套件）
from ta.volatility import AverageTrueRange
from ta.momentum import StochasticOscillator

class EnhancedTradingBot(TradingBot):
    def calculate_volatility(self, data):
        """计算波动率指标"""
        atr = AverageTrueRange(data['High'], data['Low'], data['Close'], window=14)
        return atr.average_true_range().iloc[-1]
    
    def generate_price_advice(self, data, signal):
        """生成价格建议"""
        latest = data.iloc[-1]
        atr = self.calculate_volatility(data)
        
        # 买价逻辑：收盘价与波动率加权
        if signal == 'BUY':
            buy_price = latest['Close'] * 0.995  # 挂低0.5%
            stop_loss = buy_price - 2 * atr      # 2倍ATR止损
            take_profit = buy_price + 3 * atr    # 3倍ATR止盈
            return buy_price, stop_loss, take_profit
        
        # 卖价逻辑：基于阻力位与随机指标
        elif signal == 'SELL':
            sto = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14)
            overbought_level = 80
            resistance = data['High'].rolling(5).max().iloc[-1]
            
            sell_price = max(latest['Close']*1.005, resistance)  # 挂高0.5%或突破阻力
            stop_loss = sell_price + 2 * atr if latest['Close'] < resistance else sell_price + atr
            take_profit = sell_price - 3 * atr
            return sell_price, stop_loss, take_profit
        
        else:
            return None, None, None

    def get_real_time_signal(self, ticker):
        """增强版实时信号生成"""
        data = yf.download(ticker, period='1mo', interval='1d')
        data = BacktestEngine()._calculate_features(data)
        
        latest = data.iloc[-1]
        features = latest[['RSI', 'MA20', 'MA5', 'Volatility', 'Volume']].values.reshape(1, -1)
        
        # 风险检查
        risk_score = self.risk_model.calculate_risk(ticker)
        news_risk = self._get_latest_news_risk(ticker)
        total_risk = max(risk_score, news_risk)
        
        if total_risk > 0.7:
            sell_price, stop_loss, take_profit = self.generate_price_advice(data, 'SELL')
            return 'SELL', 0.0, sell_price, stop_loss, take_profit
        
        # 机器学习预测
        model = HybridModel()
        prob = model.predict(features)[0]
        
        if prob > 0.6:
            buy_price, stop_loss, take_profit = self.generate_price_advice(data, 'BUY')
            return 'BUY', prob, buy_price, stop_loss, take_profit
        elif prob < 0.4:
            sell_price, stop_loss, take_profit = self.generate_price_advice(data, 'SELL')
            return 'SELL', prob, sell_price, stop_loss, take_profit
        else:
            return 'HOLD', prob, None, None, None

class EnhancedStrategy(AdvancedStrategy):
    def next(self):
        ticker = self.data._name
        current_price = self.data.Close[-1]
        risk_score = self.risk_model.calculate_risk(ticker)
        
        # 动态仓位调整（基于风险评分）
        max_position_size = 0.2 if risk_score < 0.3 else 0.1  # 高风险股票限仓
        
        # 信号生成
        features = pd.DataFrame({
            'RSI': [self.data.RSI[-1]],
            'MA20': [self.data.MA20[-1]],
            'MA5': [self.data.MA5[-1]],
            'Volatility': [self.data.Volatility[-1]],
            'Volume': [self.data.Volume[-1]]
        })
        prob = self.ml_model.predict(features)[0]
        
        # 交易逻辑
        if prob > 0.65 and not self.position:
            # 计算买入参数
            atr = self.calculate_volatility(self.data.df)
            buy_price = current_price * 0.995
            stop_loss = buy_price - 2 * atr
            take_profit = buy_price + 3 * atr
            
            # 执行买入并设置止损
            self.buy(
                size=min(max_position_size*self.equity//buy_price, 
                        self.equity//buy_price),
                sl=stop_loss,
                tp=take_profit
            )
            
        elif prob < 0.35 and self.position:
            self.sell()
            
        # 移动止损逻辑
        if self.position and self.position.size > 0:
            if current_price > self.position.entry_price * 1.1:
                # 盈利超过10%时启动追踪止损
                new_stop = max(self.position.sl, current_price - atr)
                self.position.update(sl=new_stop)

    def calculate_volatility(self, data):
        """ATR波动率计算"""
        return AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range().iloc[-1]

# ========== 主程式更新 ==========
if __name__ == "__main__":
    DataEngine().update_sanctions_list()
    bot = EnhancedTradingBot()
    results = []
    
    for ticker in Config.STOCK_LIST:
        try:
            # 实时信号与价格建议
            signal, confidence, price, sl, tp = bot.get_real_time_signal(ticker)
            
            # 回测验证
            stats = BacktestEngine().run_backtest(ticker)
            win_rate = stats['Win Rate [%]']/100
            sharpe = stats['Sharpe Ratio']
            max_dd = stats['Max. Drawdown [%]']/100
            
            # 风险评分
            risk_score = bot.risk_model.calculate_risk(ticker)
            
            # 收集结果
            result = {
                'Ticker': ticker,
                'Signal': signal,
                'Confidence': f"{confidence:.2f}",
                'Price_Advice': price if price else '-',
                'Stop_Loss': sl if sl else '-',
                'Take_Profit': tp if tp else '-',
                'Hist_Win_Rate': f"{win_rate:.2%}",
                'Max_Drawdown': f"{max_dd:.2%}",
                'Risk_Score': f"{risk_score:.2f}",
                'Sharpe': f"{sharpe:.2f}"
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    # 输出结果
    result_df = pd.DataFrame(results)
    print("\n增强版交易信号与价格建议：")
    print(result_df.to_markdown(index=False, floatfmt=".2f"))
