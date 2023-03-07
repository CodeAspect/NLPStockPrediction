""" 
RL Code from:

https://github.com/letianzj/QuantResearch/blob/master/ml/reinforcement_trader.ipynb

"""
import os
import io
import tempfile
import shutil
import zipfile

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import gym
import quanttrader as qt
from quanttrader import TradingEnv
import pyfolio as pf

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential, q_network, network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import pandas as pd
import json
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def flatten_tweet(t):
    d = dict()
    d['created_at'] = t['created_at']
    d['id'] = t['id']
    d['like_count'] = t['public_metrics']['like_count']
    d['quote_count'] = t['public_metrics']['quote_count']
    d['reply_count'] = t['public_metrics']['reply_count']
    d['retweet_count'] = t['public_metrics']['retweet_count']
    d['text'] = t['text']
    return d

def make_twitter_dataset():
    files = ['aantonop.json','bloodgoodBTC.json',
             'cryptocom.json','DocumentingBTC.json','HsakaTrades.json',
             'TrustWallet.json','APompliano.json','brian_armstrong.json',
             'crypto.json','elliotrades.json','whale_alert.json',
             'BitcoinMagazine.json','BTC_Archive.json','BTCTN.json','danheld.json','ForbesCrypto.json','TheCryptoDog.json'
    ]
    datadict = []
    """
    {'created_at': '2021-12-19T03:48:57.000Z', 'id': '1472413617463205893', 'public_metrics': {'like_count': 23, 'quote_count': 0, 'reply_count': 1, 'retweet_count': 0}, 'text': '@ThinkingBitmex https://t.co/ACcy7XPQCx'}
    
    """
    for f in files:
        with open("crypto/json/" + f) as ff:
            data = json.load(ff)
            for l in data:
                datadict += list(map(flatten_tweet, l))
    
    sid = SentimentIntensityAnalyzer()
    btc_tweets = pd.DataFrame(datadict)
    btc_tweets.created_at = pd.to_datetime(btc_tweets.created_at).dt.date
    btc_tweets['scores'] = btc_tweets['text'].apply(lambda review: sid.polarity_scores(review))
    btc_tweets['compound']  = btc_tweets['scores'].apply(lambda score_dict: score_dict['compound'])
    btc_tweets['comp_score'] = btc_tweets['compound'].apply(lambda c: 1 if c >=0 else -1)
    btc_tweets.drop(columns=['id'], inplace=True)
    
    return btc_tweets.groupby('created_at').mean()

def make_socialanimal_dataset():
    """ """
    sid = SentimentIntensityAnalyzer()
    btc_sa_idx = ['created_at', 'title', 'total_share_count', 'word_count',
            'twitter_shares', 'facebook_shares', 'reddit_shares', 'pinterest_shares']
    btc_sa = pd.read_csv('crypto/bitcoin-2022-03-22-SocialAnimal.csv')
    btc_sa2 = btc_sa[btc_sa_idx]
    btc_sa2.created_at = pd.DataFrame({'Date' : pd.to_datetime(btc_sa2['created_at']).dt.date})
    btc_sa2['scores'] = btc_sa2['title'].apply(lambda review: sid.polarity_scores(review))
    btc_sa2['compound']  = btc_sa2['scores'].apply(lambda score_dict: score_dict['compound'])
    btc_sa2['comp_score'] = btc_sa2['compound'].apply(lambda c: 1 if c >=0 else -1)
    return btc_sa2.groupby('created_at').mean()


def make_technical_dataset(btc_merged):
   """ """
   sym='btc'
   import ta
   df = pd.read_json('./crypto/btcdaily.json')
   if btc_merged:
       df = df.join(btc_merged)
   df = df.reindex(index=df.index[::-1])
   

   df_obs = pd.DataFrame()             # observation
   df_exch = pd.DataFrame()            # exchange; for order match

   df_exch = pd.concat([df_exch, df['close'].rename(sym)], axis=1)
   df.columns = [f'{sym}:{c.lower()}' for c in df.columns]


   macd = ta.trend.MACD(close=df[f'{sym}:close'])
   df[f'{sym}:macd'] = macd.macd()
   df[f'{sym}:macd_diff'] = macd.macd_diff()
   df[f'{sym}:macd_signal'] = macd.macd_signal()

   rsi = ta.momentum.RSIIndicator(close=df[f'{sym}:close'])
   df[f'{sym}:rsi'] = rsi.rsi()

   bb = ta.volatility.BollingerBands(close=df[f'{sym}:close'], window=20, window_dev=2)
   df[f'{sym}:bb_bbm'] = bb.bollinger_mavg()
   df[f'{sym}:bb_bbh'] = bb.bollinger_hband()
   df[f'{sym}:bb_bbl'] = bb.bollinger_lband()

   atr = ta.volatility.AverageTrueRange(high=df[f'{sym}:high'], low=df[f'{sym}:low'], close=df[f'{sym}:close'])
   df[f'{sym}:atr'] = atr.average_true_range()
   

   df_obs = pd.concat([df_obs, df], axis=1)

   print(df_obs, df_exch)
   return df_obs, df_exch
   # return btc_tech
   

def make_dataset(withtext=False):
    btc_sa = make_socialanimal_dataset()
    btc_twit = make_twitter_dataset()
    btc_sa.reset_index(inplace=True)
    btc_twit.reset_index(inplace=True)
    btc_merged = pd.merge(btc_sa, btc_twit, on='created_at',
                          how='outer')
    btc_merged.fillna(0.0, inplace=True) 
    btc_merged.set_index('created_at', inplace=True)
    if withtext:
        df_obs, df_exch = make_technical_dataset(btc_merged)
    else:
        df_obs, df_exch = make_technical_dataset(None)
    df_exch.fillna(0.0, inplace=True)
    df_obs.fillna(0.0, inplace=True)
    return df_obs, df_exch


####
# Helpers
####

def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def create_policy_eval_video(env, policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = env.reset()
      video.append_data(env.pyenv.envs[0].render())

      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        video.append_data(env.pyenv.envs[0].render())

  return embed_mp4(filename)

####
# END Helpers
####

def make_trader(df_obs, df_exch):
    """
    TODO: make and test traders with and without
    sentiment to see which performs better.
    """
    print("df_obs\n", df_obs)
    print("df_exch\n", df_exch)

    look_back = 10
    cash = 100_000.0
    max_nav_scaler = cash

    train_qt_env = TradingEnv(2, df_obs, df_exch)
    train_qt_env.set_cash(cash)
    train_qt_env.set_commission(0.0001)
    train_qt_env.set_steps(n_lookback=10, n_warmup=50, n_maxsteps=100)
    train_qt_env.set_feature_scaling(max_nav_scaler)

    eval_qt_env = TradingEnv(2, df_obs, df_exch)
    eval_qt_env.set_cash(cash)
    eval_qt_env.set_commission(0.0001)
    eval_qt_env.set_steps(n_lookback=10, n_warmup=50, n_maxsteps=2000, n_init_step=204)         # index 504 is 2012-01-03
    eval_qt_env.set_feature_scaling(max_nav_scaler)

    train_qt_env = gym.wrappers.FlattenObservation(train_qt_env)
    train_py_env = suite_gym.wrap_env(train_qt_env)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    eval_qt_env = gym.wrappers.FlattenObservation(eval_qt_env)
    eval_py_env = suite_gym.wrap_env(eval_qt_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    
    learning_rate = 1e-3 
    num_eval_episodes = 10
    replay_buffer_max_length = 100000

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    
    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    print("DQN Summary")
    print(q_net.summary())
    
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    print(agent.collect_data_spec)
    
    print("Here -2")
    replay_buffer = TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)
    print("Here -1")
    train_env.reset()

    print("Here 0")
    print("Replay Buffer", replay_buffer)
    trajectories, buffer_info = replay_buffer.get_next(sample_batch_size=2, num_steps=3) 
    print("Here 1")
    from tf_agents.trajectories.trajectory import to_transition
    time_steps, action_steps, next_time_steps = to_transition(trajectories)

    print("Here 2")
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2).prefetch(3)
    print("Here 3")
    iterator = iter(dataset)
    print("Here 4")
    num_iterations = 1_000_000   # less intelligence, more persistance; 24x7 player
    save_interval = 100_000
    eval_interval = 50_000
    log_interval = 5_000

    # Create a driver to collect experience.
    collect_driver = DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=4) # collect 4 steps for each training iteration

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    collect_driver.run = common.function(collect_driver.run)
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(train_env, agent.policy, num_episodes=1)[0]
    returns = np.array([avg_return])

    # Reset the environment.
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    
    num_iterations = 10_000
    while True:
        # Collect a few steps using collect_policy and save to the replay buffer.
        time_step, policy_state = collect_driver.run(time_step, policy_state)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()
        print(f'\r step {step}', end='')

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(train_env, agent.policy, num_episodes=1)[0]
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns = np.append(returns, avg_return)

        # if step % save_interval == 0:
        #     save_checkpoint_to_local()

        if step > num_iterations:
            break
    

    tempdir = './crypto' # os.getenv("TEST_TMPDIR", tempfile.gettempdir())

    policy_dir = os.path.join(tempdir, 'policy_btc_full')
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    

    tf_policy_saver.save(policy_dir)
    policy_zip_filename = create_zip_file(policy_dir, os.path.join(tempdir, 'exported_policy_btc_full'))
    checkpoint_dir = os.path.join(tempdir, 'checkpoint')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step_counter)
    train_checkpointer.save(train_step_counter)
    checkpoint_zip_filename = create_zip_file(checkpoint_dir, os.path.join(tempdir, 'exported_cp_btc_full'))
    create_policy_eval_video(eval_env, agent.policy, "trained-agent-btc-full", num_episodes=1)


def create_zip_file(dirname, base_filename):
    return shutil.make_archive(base_filename, 'zip', dirname)

def compute_avg_return(environment, policy, num_episodes=5):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0
    zeros = 0
    ones = 0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      if action_step.action.numpy()[0] == 1:
        ones+=1
      else:
        zeros+=1
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0], zeros, ones
 
# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

def load_stock_data(syms = ['AAPL', 'AMZN']): # ['HYSR', 'AMD', 'AMC', 'GME']):
    
    from datetime import timedelta
    import ta

    start_date = datetime(2010, 1, 1)
    end_date = datetime(2022, 3, 28)
    max_price_scaler = 5_000.0
    max_price_scaler = 1
    max_volume_scaler = 1.5e8
    df_obs = pd.DataFrame()             # observation
    df_exch = pd.DataFrame()            # exchange; for order match

    for sym in syms:

        ### XXX: Ideally, we would be using AlphaVantage API instead of yahoo finance, however, YF appears to provide more data..
        df = yf.download(sym, start=start_date, end=end_date)
        df.index = pd.to_datetime(df.index) + timedelta(hours=15, minutes=59, seconds=59)

        df_exch = pd.concat([df_exch, df['Close'].rename(sym)], axis=1)

        df['Open'] = df['Adj Close'] / df['Close'] * df['Open'] / max_price_scaler
        df['High'] = df['Adj Close'] / df['Close'] * df['High'] / max_price_scaler
        df['Low'] = df['Adj Close'] / df['Close'] * df['Low'] / max_price_scaler
        df['Volume'] = df['Adj Close'] / df['Close'] * df['Volume'] / max_volume_scaler
        df['Close'] = df['Adj Close'] / max_price_scaler
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = [f'{sym}:{c.lower()}' for c in df.columns]

        macd = ta.trend.MACD(close=df[f'{sym}:close'])
        df[f'{sym}:macd'] = macd.macd()
        df[f'{sym}:macd_diff'] = macd.macd_diff()
        df[f'{sym}:macd_signal'] = macd.macd_signal()

        rsi = ta.momentum.RSIIndicator(close=df[f'{sym}:close'])
        df[f'{sym}:rsi'] = rsi.rsi()

        bb = ta.volatility.BollingerBands(close=df[f'{sym}:close'], window=20, window_dev=2)
        df[f'{sym}:bb_bbm'] = bb.bollinger_mavg()
        df[f'{sym}:bb_bbh'] = bb.bollinger_hband()
        df[f'{sym}:bb_bbl'] = bb.bollinger_lband()

        atr = ta.volatility.AverageTrueRange(high=df[f'{sym}:high'], low=df[f'{sym}:low'], close=df[f'{sym}:close'])
        df[f'{sym}:atr'] = atr.average_true_range()

        df_obs = pd.concat([df_obs, df], axis=1)
    return df_obs, df_exch

def main():
    df_obs, df_exch = make_dataset(withtext=False)
    print(df_exch, df_obs) 
    make_trader(df_obs, df_exch)
  
if __name__ == '__main__':
    main()
