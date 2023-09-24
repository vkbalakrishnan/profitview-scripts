# Note: this file is to be used within profitview.net/trading/bots

from profitview import Link, http, logger

import json
import numpy as np
import pandas as pd
import scipy
import talib
from scipy.interpolate import interp1d
from talib import RSI
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
				'grid_size': 20000,
				'candles': {},
				'tob': (np.nan, np.nan),
				'max_risk': 100000,
				'current_risk': 0
			}, 
			'ETHUSDT' : {
				'sym': 'ETHUSDT',
				'grid_size': 10000,
				'candles': {},
				'tob': (np.nan, np.nan),
				'max_risk': 50000,
				'current_risk': 0
			} 
			# 'SOLUSDT' : {
			# 	'sym': 'SOLUSDT',
			# 	'grid_size': 10000,
			# 	'candles': {},
			# 	'tob': (np.nan, np.nan),
			# 	'max_risk': 100000,
			# 	'current_risk': 0
			# }
		}                          # symbol we will trade
        self.on_start()	
    
	def on_start(self):
		for sym in self.sym:
    		candles = self.fetch_candles(self.venue, sym, level='1m')
			# logger.info(sym)
			# logger.info('\n' + json.dumps(self.sym[sym]))
			# logger.info('\n' + json.dumps(self.sym[sym]['candles']))
			self.sym[sym]['candles'] = {x['time']: x['close'] for x in candles['data']} | self.sym[sym]['candles']	
    	self.minutely_update()
		
	def hypo_rsi(self, closes, ret):
		return RSI(np.append(closes, [closes[-1] * (1 + ret)]))[-1]
	
	def minutely_update(self):
		self.fetch_current_risk()
		self.update_limit_orders()
        threading.Timer(61 - self.second, self.minutely_update).start()
        
    def fetch_current_risk(self):
       for x in self.fetch_positions(self.venue)['data']:
			if x['sym'] in self.sym:
            	self.sym[x['sym']]['current_risk'] = x['pos_size']
  

    def orders_intent(self, sym):
		tob_bid, tob_ask = sym['tob']
		times, closes = zip(*sorted(sym['candles'].items())[-100:])
		closes = list(filter(None,closes))
		X = np.linspace(-0.2, 0.2, 100)
		Y = [self.hypo_rsi(closes, x) for x in X]
		func = interp1d(Y, X, kind='cubic', fill_value='extrapolate')
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
			'bids': [np.min([tob_bid, (0.5 * round(closes[-1] * (1 + float(func(x))) / 0.5))]) for x in (40, 30, 20, 10)],
			'asks': [np.max([tob_ask, (0.5 * round(closes[-1] * (1 + float(func(x))) / 0.5))]) for x in (60, 70, 80, 90)]
		}
		logger.info('\n'+ sym['sym'] + ':'+ json.dumps(orders))
		return orders
    
	def create_order(self, venue, sym, side, size, price):
		insert = {'symbol': sym, 'side': side, 'orderQty': size, 'price': price}
		response = self.call_endpoint(
			venue, 
			'order', 
			'private', 
			method='POST', params={
				**insert,
				'ordType': 'Limit',
				'execInst': 'ParticipateDoNotInitiate'
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
			if(abs(self.sym[sym]['current_risk']) < self.sym[sym]['max_risk'] or self.sym[sym]['current_risk'] <= 0):
				# If we are at risk limits, place larger orders 
				multiplyer = 1
			if(abs(self.sym[sym]['current_risk']) >= self.sym[sym]['max_risk']):
				multiplyer = 2
			for bid in bids:
				self.create_order(self.venue, sym=sym, side='Buy', size=self.sym[sym]['grid_size'], price=bid)		               
			if(abs(self.sym[sym]['current_risk']) < self.sym[sym]['max_risk'] or self.sym[sym]['current_risk'] >= 0):
				# If we are at risk limits, place larger orders 
				multiplyer = 1
			if(abs(self.sym[sym]['current_risk']) >= self.sym[sym]['max_risk']):
				multiplyer = 2
			for ask in asks:
				self.create_order(self.venue, sym=sym, side='Sell', size=self.sym[sym]['grid_size'], price=ask)   

			time.sleep(5)
			logger.info('\n' + json.dumps(log_msg))
            

    def trade_update(self, src, sym, data):
		if sym in self.sym:
			self.sym[sym]['candles'][self.candle_bin(data['time'], '1m')] = data['price']
			
	def quote_update(self, src, sym, data):
        if sym in self.sym:
            self.sym[sym]['tob'] = (data['bid'][0], data['ask'][0])
    