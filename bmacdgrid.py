# pylint: disable=locally-disabled, import-self, import-error, missing-class-docstring, invalid-name, consider-using-dict-items

from profitview import Link, http, logger

import json
import numpy as np
import pandas as pd
import scipy
import talib
from scipy.interpolate import interp1d
from talib import RSI
import asyncio
import time

#MODES: LONG_TAKER | REDUCE_MODE | SHORT_TAKER | GRID
initial = {
	'candles': {},
	'highs': {},
	'lows': {},
	'tob': (np.nan, np.nan),
	'liq_price': np.nan,
	'macd': np.nan,
	'rsi': np.nan,
	'atr': np.nan,
	'longce': np.nan,
	'shortce': np.nan,
	'mode': 'GRID',
	'direction': 'FLAT'
}
RISK_MULTIPLIER =30
REDUCE_SIZE_MULTIPLIER = 6
STOP_MULTIPLIER = 6
class Trading(Link):
	ENABLE_STOPS = False
	ENABLE_LIQ_PREVENTION = False
	USE_MARKET_ORDERS = False
	LIQ_THRESHOLD = 0.05
	UPDATE_SECONDS = 60
	MULTIPLIER = 1
	SHUT_IT_DOWN = False
	GRACEFUL_SHUTDOWN = True
	GRID_BIDS = (40, 30) #(40, 35, 28)
	GRID_ASKS = (60, 70) #(60, 65, 72)
	RSI_MIN = 30
	RSI_MAX = 70
	SRC = 'bitmex'                         # exchange name as in the Glossary
	VENUES = {
		'BitMEX': {
			'XBTUSDT' : {
				'sym': 'XBTUSDT',
				'grid_size': 10000,
				'max_risk': 10000*RISK_MULTIPLIER,
				'current_risk': 0,
				'current_entry': 0,
				'price_precision': 0.5,
				'price_decimals': 1,
				'stop_atr': 15,
				**initial,
			},
			'ETHUSDT' : {
				'sym': 'ETHUSDT',
				'grid_size': 10000,
				'max_risk': 10000*RISK_MULTIPLIER,
				'current_risk': 0,
				'current_entry': 0,
				'liq_price': np.nan,
				'price_precision': 0.05,
				'price_decimals': 2,
				'stop_atr': 0.8,
				**initial,
			},
			# 'SOLUSDT' : {
			# 	'sym': 'SOLUSDT',
			# 	'grid_size': 20000,
			# 	'max_risk': 20000*RISK_MULTIPLIER,
			# 	'current_risk': 0,
			# 	'current_entry': 0,
			# 	'price_precision': 0.01,
			# 	'price_decimals': 2,
			# 	**initial
			# }
		},
		'BitMEX-Pro': {
			'XBTUSDT' : {
				'sym': 'XBTUSDT',
				'grid_size': 10000,
				'max_risk': 10000*RISK_MULTIPLIER,
				'current_risk': 0,
				'current_entry': 0,
				'price_precision': 0.5,
				'price_decimals': 1,
				'stop_atr': 15,
				**initial,
			},
			'ETHUSDT' : {
				'sym': 'ETHUSDT',
				'grid_size': 10000,
				'max_risk': 10000*RISK_MULTIPLIER,
				'current_risk': 0,
				'current_entry': 0,
				'price_precision': 0.05,
				'price_decimals': 2,
				'stop_atr': 0.8,
				**initial,
			},
			# 'SOLUSDT' : {
			# 	'sym': 'SOLUSDT',
			# 	'grid_size': 10000,
			# 	'max_risk': 10000*RISK_MULTIPLIER,
			# 	'current_risk': 0,
			# 	'current_entry': 0,
			# 	'price_precision': 0.01,
			# 	'price_decimals': 2,
			# 	**initial
			# }
		}
	}

	def __init__(self):
		super().__init__()
		self.on_start()

	def on_start(self):
		for venue in self.VENUES:
			for sym in self.VENUES[venue]:
				candles = self.fetch_candles(venue, sym, level='1m')
				self.VENUES[venue][sym]['candles'] = {x['time']: x['close'] for x in candles['data']} | self.VENUES[venue][sym]['candles']
				self.VENUES[venue][sym]['highs'] = {x['time']: x['high'] for x in candles['data']} | self.VENUES[venue][sym]['highs']
				self.VENUES[venue][sym]['lows'] = {x['time']: x['low'] for x in candles['data']} | self.VENUES[venue][sym]['lows']
		asyncio.run(self.minutely_update())

	def hypo_rsi(self, closes, ret):
		return RSI(np.append(closes, [closes[-1] * (1 + ret)]))[-1]

	async def minutely_update(self):
		while True :
			# try:
			self.fetch_current_risk()
			await self.trade()
			await asyncio.sleep(self.UPDATE_SECONDS)
			# except Exception as e:
			# 	logger.info('Failed to run trade cycle')
			# 	logger.info(e)
			# 	await asyncio.sleep(5)

	@property
	def time_bin_now(self):
		return self.candle_bin(self.epoch_now, '1m')

	def last(self, sym, metric):
		start_time = self.time_bin_now - 100 * 60_000
		times = [start_time + (i + 1) * 60_000 for i in range(100)]
		values = [sym[metric].get(x, np.nan) for x in times]
		return np.array(pd.Series(values).ffill())

	def last_closes(self, sym):
		return self.last(sym, 'candles')

	def log_current_risk(self, venue):
		current_risk = 0
		current_risk_type = 'XBT'

		for sym in self.VENUES[venue]:
			if sym.endswith('USD') :
				current_risk = self.VENUES[venue][sym]['current_risk']
				current_risk_type = 'XBT'
			else:
				current_risk = self.VENUES[venue][sym]['current_risk']
				current_risk_type = 'USDT'

			logger.info(f'\n{venue} - {sym} current risk: {current_risk} {current_risk_type}')

	def fetch_current_risk(self):
		for venue in self.VENUES:
			try:
				positions=self.fetch_positions(venue)
				if positions :
					for x in positions['data']:
						if x['sym'] in self.VENUES[venue]:
							self.VENUES[venue][x['sym']]['current_risk'] = x['pos_size']
							self.VENUES[venue][x['sym']]['risk'] = x
				# self.log_current_risk(venue)
			except Exception as e:
				logger.info('Failed to fetch positions')
			

	def remove_duplicates(self, arr):
		unique_items = list(set(arr))
		return unique_items

	def round_value(self, x, tick, decimals=0):
		return np.round(tick * np.round(x / tick), decimals)

	def orders_intent(self, sym):
		tob_bid, tob_ask = sym['tob']
		times, closes = zip(*sorted(sym['candles'].items())[-100:])
		closes = list(filter(None,closes))
		X = np.linspace(-0.2, 0.2, 100)
		Y = [self.hypo_rsi(closes, x) for x in X]
		func = interp1d(Y, X, kind='cubic', fill_value='extrapolate')

		orders = {
			'bids': [np.min([tob_bid, self.round_value(0.5 * round(closes[-1] * (1 + float(func(x))) / 0.5,4),sym['price_precision'], sym['price_decimals'])]) for x in self.GRID_BIDS],
			'asks': [np.max([tob_ask, self.round_value(0.5 * round(closes[-1] * (1 + float(func(x))) / 0.5,4),sym['price_precision'], sym['price_decimals'])]) for x in self.GRID_ASKS]
		}
		orders['bids'] = self.remove_duplicates(orders['bids'])
		orders['asks'] = self.remove_duplicates(orders['asks'])

		logger.info('sym:' + sym['sym'] + json.dumps(orders))
		return orders

	def update_signal(self, sym):
		closes = self.last_closes(sym);
		highs = self.last(sym, 'highs');
		lows = self.last(sym, 'lows');
		macd, signal, hist = talib.MACD(closes)
		sym['macd'] = hist[-1]
		sym['rsi'] = talib.RSI(closes, timeperiod=14)[-1]
		atr = talib.ATR(highs, lows, closes, 14)
		# atr = talib.ATR(np.array(highs, dtype=np.float64), np.array(lows, dtype=np.float64), np.array(closes, dtype=np.float64), 30)
		sym['atr'] = atr[-1]
		sym['longce'] = np.max(highs[-30:]) - STOP_MULTIPLIER*atr[-1]
		sym['shortce'] = np.min(lows[-30:]) + STOP_MULTIPLIER*atr[-1]
		# prevlongce = np.max(highs[-23:][:-1]) - STOP_MULTIPLIER*atr[-2]
		# prevshortce = np.max(lows[-23:][:-1]) + STOP_MULTIPLIER*atr[-2]

		# logger.info('\n'+json.dumps({
		# 	'nan': 'lonce' in sym,
		# 	'atr': atr[-1],
		# 	'longce': self.round_value(sym['longce'], sym['price_precision'], sym['price_decimals']),
		# 	'shortce': self.round_value(sym['shortce'], sym['price_precision'], sym['price_decimals'])
		# }))

		# df = {}
		# #  Long position
		# df['enter_long'] = 1 if (closes[-1] > sym['shortce']) & (closes[-2] <= prevshortce) else 0
		# df['exit_long'] = 1 if (closes[-1] < sym['longce']) & (closes[-2] >= prevlongce) else 0

		# #  Short position
		# df['enter_short'] = 1 if (closes[-1] < sym['longce']) & (closes[-2] >= prevlongce) else 0
		# df['exit_short'] = 1 if (closes[-1] > sym['shortce']) & (closes[-2] <= prevshortce) else 0
		# logger.info({ **df, 'signal': sum(df.values())})
		# # logger.info(sum(df.values()))

		
		
	def compute_mode(self, sym):
		previous_macd = sym['macd']
		previous_rsi = sym['rsi']
		self.update_signal(sym)
		# logger.info('MACD prev: ' + str(previous_macd) + ' curr: ' + str(sym['macd']))
		# logger.info('RSI prev: ' + str(previous_rsi) + ' curr: ' + str(sym['rsi']))
		current_price = self.last_closes(sym)[-1];
		liq_threat = 1
		sym['stop_price'] = 0
		stop_price = 0
		
		if(np.isnan(previous_macd) or np.isnan(previous_rsi)):
			return

		if "risk" in sym and "side" in  sym["risk"]:
			logger.info(
				f'| Entry: {sym["risk"]["entry_price"]}'
			);	
			if sym["risk"].get("side") == 'Buy':
				liq_threat = (current_price - sym["risk"]["liq_price"]) / sym["risk"]["entry_price"]
				stop_price = 0 if np.isnan(sym["longce"]) else sym["longce"];
			elif sym["risk"].get("side") == 'Sell':
				liq_threat = (sym["risk"]["liq_price"] - current_price) / sym["risk"]["entry_price"]
				stop_price = 0 if np.isnan(sym["shortce"]) else sym["shortce"];



		sym['stop_price'] = self.round_value(stop_price, sym['price_precision'], sym['price_decimals'])
		logger.info(
			f'Current {sym["sym"]} price: {str(current_price)}'+
			f'| Threat: {liq_threat}'
		);	
		
		if (sym['current_risk'] == 0 and abs(current_price- stop_price) < sym['atr']):
			logger.info(f'current risk: {sym["current_risk"]} | current: {current_price} | stop price: {stop_price} | diff {abs(current_price- stop_price)}')
			sym['mode'] = 'SKIP'
			return
			
		self.USE_MARKET_ORDERS = False
		if(self.ENABLE_LIQ_PREVENTION is True and liq_threat < self.LIQ_THRESHOLD):
			self.USE_MARKET_ORDERS = True
			if sym["risk"]["side"] == 'Buy':
				sym['mode'] = 'REDUCE_SHORT'
			else:
				sym['mode'] = 'REDUCE_LONG'
		elif(np.greater(sym['rsi'], self.RSI_MAX) and sym['macd'] > previous_macd):
			sym['mode'] = 'TAKER_LONG'
			# sym['mode'] = 'TAKER_SHORT'
		elif(np.greater(self.RSI_MIN, sym['rsi']) and sym['macd'] < previous_macd):
			sym['mode'] = 'TAKER_SHORT'
			# sym['mode'] = 'TAKER_LONG'
		elif(sym['mode'] == 'TAKER_LONG' and sym['macd'] > previous_macd):
			# sym['mode'] = 'GRID'
			sym['mode'] = 'REDUCE_SHORT'
		elif(sym['mode'] == 'TAKER_SHORT' and sym['macd'] < previous_macd):
			# sym['mode'] = 'GRID'
			sym['mode'] = 'REDUCE_LONG'
		elif((sym['mode'] == 'REDUCE_SHORT' or sym['mode'] == 'REDUCE_LONG') and (abs(sym['current_risk']) <= sym['grid_size'] or (sym['rsi'] > self.RSI_MIN and sym['rsi'] < self.RSI_MAX))):
			sym['mode'] = 'GRID'


	def call_post_order_endpoint(self, venue, params):
		self.call_endpoint(
			venue,
			'order',
			'private',
			method='POST', 
			params= {
				**params,
				'text': 'Sent from ProfitView.net'
			}
		)
		self.last_order_at = time.time()

	async def trade(self):
		for venue in self.VENUES:
			for sym in self.VENUES[venue]:
				self.compute_mode(self.VENUES[venue][sym])
				logger.info("Symbol: "+sym+" | Current Mode: " + self.VENUES[venue][sym]['mode']+" | Current Risk: "+ str(self.VENUES[venue][sym]['current_risk']) )
				logger.info(f'Stop order at {self.VENUES[venue][sym]["stop_price"]} | ATR: {self.VENUES[venue][sym]["atr"]} ({self.VENUES[venue][sym]["stop_atr"]}) ')
				
				if(self.VENUES[venue][sym]['mode'] == 'SKIP'): 
					logger.info(f'Skipping {sym}')
					return
						
				if(self.VENUES[venue][sym]['mode'] == 'GRID'):
					await self.update_limit_orders(venue, sym)
					await asyncio.sleep(3)
				elif(self.VENUES[venue][sym]['mode'] == 'TAKER_LONG'):
					await self.taker_long_orders(venue, sym)
				elif(self.VENUES[venue][sym]['mode'] == 'TAKER_SHORT'):
					await self.taker_short_orders(venue, sym)
				elif(self.VENUES[venue][sym]['mode'] == 'REDUCE_SHORT'):
					await self.taker_reduce_short_orders(venue, sym)
				elif(self.VENUES[venue][sym]['mode'] == 'REDUCE_LONG'):
					await self.taker_reduce_long_orders(venue, sym)
				# elif(self.VENUES[venue][sym]['mode'] == 'STOP_LONG'):
				# 	await self.stop_order(venue, sym, 'Sell')
				# elif(self.VENUES[venue][sym]['mode'] == 'STOP_SHORT'):
				# 	await self.stop_order(venue, sym, 'Buy')
				
				
				if self.ENABLE_STOPS is True and self.VENUES[venue][sym]["atr"]>self.VENUES[venue][sym]["stop_atr"] and self.VENUES[venue][sym]["stop_price"] > 0:
					try:
						await self.stop_order(venue, sym, self.VENUES[venue][sym]['current_risk'])
					except Exception as e:
						logger.info('failed to set stop order')
						logger.error(e)
				
	async def stop_order(self, venue, sym, risk):
		if risk == 0 or np.isnan(self.VENUES[venue][sym]['stop_price']) or self.VENUES[venue][sym]['stop_price'] == 0:
			return;
		side = 'Buy' if risk < 0 else 'Sell'
		self.call_post_order_endpoint(
				venue,
				params={
					'symbol': sym,
					'side': side,
					'ordType': 'Stop',
					'execInst': 'Close,LastPrice',
					'orderQty': abs(risk),
					'stopPx': self.VENUES[venue][sym]['stop_price']
				}
			)

	async def taker_reduce_long_orders(self, venue, sym): 
		tob_bid, tob_ask = self.VENUES[venue][sym]['tob']
		if(np.isnan(tob_bid) or np.isnan(tob_ask)):
			return
		self.cancel_order(venue, sym=sym)

		if (self.VENUES[venue][sym]['current_risk'] <= 0):
			# If taker_long, then market buy one and place a tob other
			if self.USE_MARKET_ORDERS is True:
				self.call_post_order_endpoint(
					venue,
					params={
						'symbol': sym,
						'side': 'Buy',
						'orderQty': self.VENUES[venue][sym]['grid_size'] * REDUCE_SIZE_MULTIPLIER,
						'ordType': 'Market',
					}
				)

			self.call_post_order_endpoint(
				venue,
				params={
					'symbol': sym,
					'side': 'Buy',
					'price': tob_bid,
					'ordType': 'Limit',
					'orderQty': self.VENUES[venue][sym]['grid_size'] * REDUCE_SIZE_MULTIPLIER,
					'execInst': 'ParticipateDoNotInitiate',
				}
			)

	async def taker_reduce_short_orders(self, venue, sym): 
		tob_bid, tob_ask = self.VENUES[venue][sym]['tob']
		if(np.isnan(tob_bid) or np.isnan(tob_ask)):
			return
		self.cancel_order(venue, sym=sym)

		if (self.VENUES[venue][sym]['current_risk'] >= 0):
			# If taker_long, then market buy one and place a tob other
			if self.USE_MARKET_ORDERS is True:
				self.call_post_order_endpoint(
					venue,
					params={
						'symbol': sym,
						'side': 'Sell',
						'orderQty': self.VENUES[venue][sym]['grid_size'] * REDUCE_SIZE_MULTIPLIER,
						'ordType': 'Market',
					}
				)

			self.call_post_order_endpoint(
				venue,
				params={
					'symbol': sym,
					'side': 'Sell',
					'price': tob_bid,
					'ordType': 'Limit',
					'orderQty': self.VENUES[venue][sym]['grid_size'] * REDUCE_SIZE_MULTIPLIER,
					'execInst': 'ParticipateDoNotInitiate',
				}
			)

	
	async def taker_long_orders(self, venue, sym, reduce = False): 
		tob_bid, tob_ask = self.VENUES[venue][sym]['tob']
		multiplier = 2
		if(np.isnan(tob_bid) or np.isnan(tob_ask)):
			return
		self.cancel_order(venue, sym=sym)
		if (abs(self.VENUES[venue][sym]['current_risk']) < self.VENUES[venue][sym]['max_risk']) or (self.VENUES[venue][sym]['current_risk'] <= 0):
			# If taker_long, then market buy one and place a tob other
			if self.USE_MARKET_ORDERS is True:
				self.call_post_order_endpoint(
					venue,
					params={
						'symbol': sym,
						'side': 'Buy',
						'orderQty': self.VENUES[venue][sym]['grid_size'] * multiplier,
						'ordType': 'Market',
					}
				)

			self.call_post_order_endpoint(
				venue,
				params={
					'symbol': sym,
					'side': 'Buy',
					'price': tob_bid,
					'ordType': 'Limit',
					'orderQty': self.VENUES[venue][sym]['grid_size'] * multiplier,
					'execInst': 'ParticipateDoNotInitiate',
				}
			)

	async def taker_short_orders(self, venue, sym): 
		tob_bid, tob_ask = self.VENUES[venue][sym]['tob']
		multiplier = 2

		if(np.isnan(tob_bid) or np.isnan(tob_ask)):
			return
		self.cancel_order(venue, sym=sym)
		if (abs(self.VENUES[venue][sym]['current_risk']) < self.VENUES[venue][sym]['max_risk']) or (self.VENUES[venue][sym]['current_risk'] >= 0):
			# If taker_short, then market buy one and place a tob other
			if self.USE_MARKET_ORDERS is True:
				self.call_post_order_endpoint(
					venue,
					params={
						'symbol': sym,
						'side': 'Sell',
						'ordType': 'Market',
						'orderQty': self.VENUES[venue][sym]['grid_size'] * multiplier,
					}
				)

			self.call_post_order_endpoint(
				venue,
				params={
					'symbol': sym,
					'side': 'Sell',
					'price': tob_ask,
					'ordType': 'Limit',
					'orderQty': self.VENUES[venue][sym]['grid_size'] * multiplier,
					'execInst': 'ParticipateDoNotInitiate',
				}
			)


	async def update_limit_orders(self, venue, sym):
		logger.info(sym)
		tob_bid, tob_ask = self.VENUES[venue][sym]['tob']
		if(np.isnan(tob_bid) or np.isnan(tob_ask)):
			return
		intent = self.orders_intent(self.VENUES[venue][sym])
		bids = intent['bids']
		asks = intent['asks']


		#cancel all current orders
		try: 
			self.cancel_order(venue, sym=sym)
		except Exception as e:
			logger.error(e)
			return
		# Buy orders
		if (abs(self.VENUES[venue][sym]['current_risk']) <  self.VENUES[venue][sym]['max_risk']) or (self.VENUES[venue][sym]['current_risk'] <= 0):
			if self.SHUT_IT_DOWN :
				if self.GRACEFUL_SHUTDOWN :
					bid = tob_bid
					execInst = 'Close,ParticipateDoNotInitiate'
				else :
					bid = tob_bid
					execInst = 'Close'

				try:
					self.call_post_order_endpoint(
						venue,
						params={
							'symbol': sym,
							'side': 'Buy',
							'price': bid,
							'ordType': 'Limit',
							'execInst': execInst,
						})
				except Exception as e:
					logger.error(e)
			else :
				# If I have a current open position in the opposite direction, double the order size
				multiplier = 1
				if(self.VENUES[venue][sym]['current_risk'] <= 0):
					multiplier = self.MULTIPLIER
				if(self.VENUES[venue][sym]['direction'] == 'SHORT'):
					execInst = 'ParticipateDoNotInitiate,ReduceOnly'
				else:
					execInst = 'ParticipateDoNotInitiate'
				for bid in bids:
					try:
						self.call_post_order_endpoint(
							venue,
							params={
								'symbol': sym,
								'side': 'Buy',
								'orderQty': self.VENUES[venue][sym]['grid_size'] * multiplier,
								'price': bid,
								'ordType': 'Limit',
								'execInst': execInst,
							})
					except Exception as e:
						logger.error(e)

		# Sell orders
		if (abs(self.VENUES[venue][sym]['current_risk']) <  self.VENUES[venue][sym]['max_risk']) or (self.VENUES[venue][sym]['current_risk'] >= 0):
			if self.SHUT_IT_DOWN :
				if self.GRACEFUL_SHUTDOWN :
					bid = tob_ask
					execInst = 'Close,ParticipateDoNotInitiate'
				else :
					bid = tob_ask
					execInst = 'Close'

				try:
					self.call_post_order_endpoint(
						venue,
						params={
							'symbol': sym,
							'side': 'Sell',
							'price': bid,
							'ordType': 'Limit',
							'execInst': execInst,
						})
				except Exception as e:
					logger.error(e)
			else :
				# If I have a current open position in the opposite direction, double the order size
				multiplier = 1
				if(self.VENUES[venue][sym]['current_risk'] >= 0):
					multiplier = self.MULTIPLIER
				if(self.VENUES[venue][sym]['direction'] == 'LONG'):
					execInst = 'ParticipateDoNotInitiate,ReduceOnly'
				else:
					execInst = 'ParticipateDoNotInitiate'
				for ask in asks:
					try:
						self.call_post_order_endpoint(
							venue,
							params={
								'symbol': sym,
								'side': 'Sell',
								'orderQty': self.VENUES[venue][sym]['grid_size'] * multiplier,
								'price': ask,
								'ordType': 'Limit',
								'execInst': execInst,
							})
					except Exception as e:
						logger.error(e)

	def trade_update(self, src, sym, data):
		try:
			for venue in self.VENUES:
				if sym in self.VENUES[venue]:
					time = self.candle_bin(data['time'], '1m')
					price = data['price']
					if time not in self.VENUES[venue][sym]['highs']:
						self.VENUES[venue][sym]['highs'][time] = price
					if time not in self.VENUES[venue][sym]['lows']:
						self.VENUES[venue][sym]['lows'][time] = price
						
					high = self.VENUES[venue][sym]['highs'][time]
					low = self.VENUES[venue][sym]['lows'][time]
					# logger.info(json.dumps({'time': time,'high': high, 'low': low}))
					self.VENUES[venue][sym]['candles'][time] = price
					self.VENUES[venue][sym]['highs'][time] = price if price > high else high
					self.VENUES[venue][sym]['lows'][time] = price if price < low else low
		except Exception as e: 
			logger.error(e)
			
	def quote_update(self, src, sym, data):
		# logger.info('\n' + json.dumps(data))
		for venue in self.VENUES:
			if sym in self.VENUES[venue]:
				self.VENUES[venue][sym]['tob'] = (data['bid'][0], data['ask'][0])
		
	@http.route
	def get_health(self, data):
		if time.time() - self.last_order_at < 180:
			return time.time() - self.last_order_at
		return Exception