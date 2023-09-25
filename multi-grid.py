# Note: this file is to be used within profitview.net/trading/bots

from profitview import Link, http, logger

import json
import numpy as np
import pandas as pd
import scipy
import talib
import threading
import time


def debounce(wait):
    """Postpone a function execution until after wait seconds
    have elapsed since the last time it was invoked. 
    source: https://gist.github.com/walkermatt/2871026"""
    def decorator(func):
        def debounced(*args, **kwargs):
            def call_func():
                debounced.last_call = time.time()
                func(*args, **kwargs)
            
            if hasattr(debounced, 'timer'):
                debounced.timer.cancel()
            
            if time.time() - getattr(debounced, 'last_call', 0) > wait:
                call_func()
            else:                
                debounced.timer = threading.Timer(wait, call_func)
                debounced.timer.start()                

        return debounced
    return decorator



class Trading(Link):
    
    def __init__(self):
        super().__init__()
        # ALGO PARAMS
        self.src = 'BitMEX'                         # exchange name
        self.venue = 'BitMEX-Pro'                         # API key name
        self.sym = {
			'XBTUSDT' : {
				'sym': 'XBTUSDT',
				'grid_size': 200000,
				'candles': {},
				'highs': {},
				'lows': {},
				'tob': (np.nan, np.nan),
				'max_risk': 1000000,
				'current_risk': 0,
				'price_precision': 0.5,
				'price_decimals': 1,
				'derisk': True
			}, 
			'ETHUSDT' : {
				'sym': 'ETHUSDT',
				'grid_size': 10000,
				'candles': {},
				'highs': {},
				'lows': {},
				'tob': (np.nan, np.nan),
				'max_risk': 250000,
				'current_risk': 0,
				'price_precision': 0.05,
				'price_decimals': 2
			}, 
			'SOLUSDT' : {
				'sym': 'SOLUSDT',
				'grid_size': 10000,
				'candles': {},
				'highs': {},
				'lows': {},
				'tob': (np.nan, np.nan),
				'max_risk': 200000,
				'current_risk': 0,
				'price_precision': 0.01,
				'price_decimals': 2
			}
		}                          # symbol we will trade
        self.on_start()	
    
	def on_start(self):
		for sym in self.sym:
    		candles = self.fetch_candles(self.venue, sym, level='1m')
			self.sym[sym]['candles'] = {x['time']: x['close'] for x in candles['data']} | self.sym[sym]['candles']	
			self.sym[sym]['highs'] = {x['time']: x['high'] for x in candles['data']} | self.sym[sym]['highs']	
			self.sym[sym]['lows'] = {x['time']: x['low'] for x in candles['data']} | self.sym[sym]['lows']	
    	self.minutely_update()
		
	def hypo_rsi(self, closes, ret):
		return talib.RSI(np.append(closes, [closes[-1] * (1 + ret)]))[-1]
	
	def stop_loss_price(self, highs, lows, closes):
		# logger.info('\n'+ sym['sym'] + ' ATR :'+ json.dumps({'highs': highs, 'lows':lows, 'closes':closes}))
		logger.info(highs)
		return talib.ATR(np.asarray(highs, dtype='float64'), np.asarray(lows, dtype='float64'), np.asarray(closes, dtype='float64'), 14)
	
	def minutely_update(self):
		self.fetch_current_risk()
		self.update_limit_orders()
        threading.Timer(61 - self.second, self.minutely_update).start()
        
    def fetch_current_risk(self):
       for x in self.fetch_positions(self.venue)['data']:
			if x['sym'] in self.sym:
            	self.sym[x['sym']]['current_risk'] = x['pos_size']
	
	def remove_duplicates(self, arr):
		unique_items = list(set(arr))
		return unique_items

	def round_value(self, x, tick, decimals=0):
		return np.round(tick * np.round(x / tick), decimals)

    def orders_intent(self, sym):
		tob_bid, tob_ask = sym['tob']
		times, closes = zip(*sorted(sym['candles'].items())[-100:])
		timesh, highs = zip(*sorted(sym['highs'].items())[-100:])
		timesl, lows = zip(*sorted(sym['lows'].items())[-100:])
		closes = list(filter(None,closes))
		highs = list(filter(None,highs))
		lows = list(filter(None,lows))
		
		# logger.info('\n'+ sym['sym'] + ' ATR :'+ self.stop_loss_price(highs, lows, closes))
		closes = list(filter(None,closes))
		X = np.linspace(-0.2, 0.2, 100)
		Y = [self.hypo_rsi(closes, x) for x in X]
		func = scipy.interpolate.interp1d(Y, X, kind='cubic', fill_value='extrapolate')
		# logger.info('\n' + json.dumps({
		# 	'sym': sym['sym'],
		# 	'total': 0.5 * round(closes[-1] * (1 + float(func(40))) / 0.5),
		# 	'close': closes[-1],
		# 	'closeLength': len(closes),
		# 	'func': (1 + float(func(40))),
		# 	'tob': tob_bid,
		# 	'final_closest_bid': np.min([tob_bid, 0.5 * round(closes[-1] * (1 + float(func(40))) / 0.5)]),
		# 	'current_risk': sym['current_risk']
		# }))
		orders = {
			'bids': [np.min([tob_bid, self.round_value(0.5 * round(closes[-1] * (1 + float(func(x))) / 0.5,2),sym['price_precision'], sym['price_decimals'])]) for x in (40, 30, 20, 10)],
			'asks': [np.max([tob_ask, self.round_value(0.5 * round(closes[-1] * (1 + float(func(x))) / 0.5,2),sym['price_precision'], sym['price_decimals'])]) for x in (60, 70, 80, 90)]
		}
		orders['bids'] = self.remove_duplicates(orders['bids'])
		orders['asks'] = self.remove_duplicates(orders['asks'])
		logger.info('\n'+ sym['sym'] + ':'+ json.dumps(orders))
		return orders
    
	def create_order(self, venue, sym, side, size, price, reduceOnly=False):
		insert = {'symbol': sym, 'side': side, 'orderQty': size, 'price': price}
		execs = ['ParticipateDoNotInitiate']
		if reduceOnly is True: 
			execs = ['ReduceOnly']
		response = self.call_endpoint(
			venue, 
			'order', 
			'private', 
			method='POST', params={
				**insert,
				'ordType': 'Limit',
				'execInst': ",".join(execs)
			}
		)
		
    @debounce(1)
    def update_limit_orders(self):
		for sym in self.sym:
			tob_bid, tob_ask = self.sym[sym]['tob']
			if(np.isnan(tob_bid) or np.isnan(tob_ask)):
				continue 
			intent = self.orders_intent(self.sym[sym])
			bids = intent['bids']
			asks = intent['asks']
			
			log_msg = {
				'bids': bids,
				'asks': asks,
			}

			#cancel all current orders
			self.cancel_order(self.venue, sym=sym)

			multiplyer = 2 if(abs(self.sym[sym]['current_risk']) >= self.sym[sym]['max_risk']) else 1
			logger.info(json.dumps({
				'risk': self.sym[sym]['current_risk'],
				'multiplyer': multiplyer,
				'reduceOnly': True if multiplyer==2 and self.sym[sym]['current_risk'] < 0 else False
			}))
			for bid in bids:
				reduceOnly= True if multiplyer==2 and self.sym[sym]['current_risk'] > 0 else False
				if reduceOnly is False:
					self.create_order(self.venue, sym=sym, side='Buy', size=self.sym[sym]['grid_size']* multiplyer, price=bid, reduceOnly=False)
			for ask in asks:
				reduceOnly= True if multiplyer==2 and self.sym[sym]['current_risk'] < 0 else False
				if reduceOnly is False:
					self.create_order(self.venue, sym=sym, side='Sell', size=self.sym[sym]['grid_size']* multiplyer, price=ask, reduceOnly=False)   

			time.sleep(5)
			logger.info('\n' + json.dumps(log_msg))
            

    def trade_update(self, src, sym, data):
		if sym in self.sym:
			self.sym[sym]['candles'][self.candle_bin(data['time'], '1m')] = data['price']
			
	def quote_update(self, src, sym, data):
        if sym in self.sym:
            self.sym[sym]['tob'] = (data['bid'][0], data['ask'][0])
    
