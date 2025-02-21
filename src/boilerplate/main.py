import json
import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from web3 import Web3
import tweepy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.publickey import PublicKey
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)
logger = logging.getLogger('PumpFun_GMGN_Bot')

class Config:
    """Configuration management class for the trading bot"""
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found. Creating default config at {self.config_path}")
            self.config = {
                "solana_rpc_url": "https://api.mainnet-beta.solana.com",
                "wallet_private_key": "",
                "priority_fee": 1.5,  # SOL
                "buy_slippage": 17,  # percentage
                "sell_slippage": 17,  # percentage
                "profit_target": 15,  # 15x target
                "long_term_percentage": 10,  # percentage to keep for long term
                "blacklisted_coins": [],
                "blacklisted_devs": [],
                "minimum_token_score": 65,
                "tweetscout_api_key": "",
                "twitter_accounts_to_monitor": [],
                "solanasniffer_api_key": "",
                "rugcheck_api_key": ""
            }
            self.save_config()
    
    def save_config(self):
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as file:
            json.dump(self.config, file, indent=4)
        logger.info(f"Configuration saved to {self.config_path}")
    
    def update_config(self, key, value):
        """Update a specific configuration value"""
        self.config[key] = value
        self.save_config()
        logger.info(f"Configuration updated: {key} = {value}")
    
    def add_to_blacklist(self, item, blacklist_type):
        """Add an item to the specified blacklist"""
        if blacklist_type == "coins":
            if item not in self.config["blacklisted_coins"]:
                self.config["blacklisted_coins"].append(item)
                self.save_config()
                logger.info(f"Added {item} to coin blacklist")
        elif blacklist_type == "devs":
            if item not in self.config["blacklisted_devs"]:
                self.config["blacklisted_devs"].append(item)
                self.save_config()
                logger.info(f"Added {item} to dev blacklist")
    
    def remove_from_blacklist(self, item, blacklist_type):
        """Remove an item from the specified blacklist"""
        if blacklist_type == "coins":
            if item in self.config["blacklisted_coins"]:
                self.config["blacklisted_coins"].remove(item)
                self.save_config()
                logger.info(f"Removed {item} from coin blacklist")
        elif blacklist_type == "devs":
            if item in self.config["blacklisted_devs"]:
                self.config["blacklisted_devs"].remove(item)
                self.save_config()
                logger.info(f"Removed {item} from dev blacklist")


class TokenDatabase:
    """Database for storing and analyzing token data"""
    def __init__(self, db_path='token_database.json'):
        self.db_path = db_path
        self.tokens = {}
        self.load_database()
    
    def load_database(self):
        """Load token database from file"""
        try:
            with open(self.db_path, 'r') as file:
                self.tokens = json.load(file)
            logger.info(f"Token database loaded with {len(self.tokens)} tokens")
        except FileNotFoundError:
            logger.warning(f"Database file not found. Creating new database at {self.db_path}")
            self.tokens = {}
            self.save_database()
    
    def save_database(self):
        """Save token database to file"""
        with open(self.db_path, 'w') as file:
            json.dump(self.tokens, file, indent=4)
        logger.info(f"Token database saved with {len(self.tokens)} tokens")
    
    def add_token(self, token_address, token_data):
        """Add or update a token in the database"""
        self.tokens[token_address] = {
            **token_data,
            "last_updated": datetime.now().isoformat()
        }
        self.save_database()
        logger.info(f"Token {token_address} added/updated in database")
    
    def get_token(self, token_address):
        """Get token data from the database"""
        return self.tokens.get(token_address)
    
    def export_to_csv(self, filepath='token_analysis.csv'):
        """Export token database to CSV for analysis"""
        if not self.tokens:
            logger.warning("No tokens in database to export")
            return False
        
        df = pd.DataFrame.from_dict(self.tokens, orient='index')
        df.to_csv(filepath)
        logger.info(f"Token database exported to {filepath}")
        return True
    
    def get_tokens_by_dev(self, dev_address):
        """Get all tokens created by a specific developer"""
        return {addr: data for addr, data in self.tokens.items() 
                if data.get('developer_address') == dev_address}
    
    def find_migration_patterns(self):
        """Analyze tokens to find migration patterns"""
        migrated_tokens = {addr: data for addr, data in self.tokens.items() 
                          if data.get('migration_data')}
        
        if not migrated_tokens:
            logger.info("No migrated tokens found for pattern analysis")
            return []
        
        patterns = []
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame.from_dict(migrated_tokens, orient='index')
        
        # Analyze time patterns
        if 'migration_timestamp' in df.columns:
            time_analysis = df['migration_timestamp'].apply(
                lambda x: datetime.fromisoformat(x) if isinstance(x, str) else x
            )
            hour_distribution = time_analysis.dt.hour.value_counts().sort_index()
            day_distribution = time_analysis.dt.dayofweek.value_counts().sort_index()
            
            patterns.append({
                'type': 'time_pattern',
                'peak_hours': hour_distribution.nlargest(3).index.tolist(),
                'peak_days': day_distribution.nlargest(3).index.tolist()
            })
        
        # Analyze performance patterns
        if all(col in df.columns for col in ['initial_price', 'peak_price']):
            df['roi'] = df['peak_price'] / df['initial_price']
            avg_roi = df['roi'].mean()
            patterns.append({
                'type': 'performance_pattern',
                'average_roi': avg_roi,
                'successful_percentage': (df['roi'] > 2).mean() * 100
            })
        
        # Developer patterns
        if 'developer_address' in df.columns:
            dev_success_rate = df.groupby('developer_address')['roi'].agg(
                ['mean', 'count']
            ).sort_values('mean', ascending=False)
            
            top_devs = dev_success_rate[dev_success_rate['count'] > 1].head(5)
            if not top_devs.empty:
                patterns.append({
                    'type': 'developer_pattern',
                    'top_developers': top_devs.index.tolist()
                })
        
        logger.info(f"Found {len(patterns)} migration patterns")
        return patterns


class PumpFunAPI:
    """API Interface for PumpFun platform"""
    def __init__(self, base_url="https://api.pumpfun.xyz/v1"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "PumpFun_Analysis_Bot/1.0",
            "Accept": "application/json"
        }
    
    def get_migrated_tokens(self, limit=100, offset=0):
        """Get list of tokens that have migrated"""
        endpoint = f"{self.base_url}/tokens/migrated"
        params = {
            "limit": limit,
            "offset": offset
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get migrated tokens: {e}")
            return None
    
    def get_token_details(self, token_address):
        """Get detailed information about a specific token"""
        endpoint = f"{self.base_url}/token/{token_address}"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get token details for {token_address}: {e}")
            return None
    
    def get_token_price_history(self, token_address, timeframe="1d"):
        """Get price history for a token"""
        endpoint = f"{self.base_url}/token/{token_address}/price"
        params = {"timeframe": timeframe}
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get price history for {token_address}: {e}")
            return None


class GMGNMonitor:
    """API Interface for GMGN platform"""
    def __init__(self, base_url="https://api.gmgn.ai/v1"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "GMGN_Analysis_Bot/1.0",
            "Accept": "application/json"
        }
    
    def get_trending_tokens(self, limit=20):
        """Get trending tokens on GMGN"""
        endpoint = f"{self.base_url}/trending"
        params = {"limit": limit}
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get trending tokens: {e}")
            return None
    
    def get_token_social_data(self, token_address):
        """Get social media metrics for a token"""
        endpoint = f"{self.base_url}/token/{token_address}/social"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get social data for {token_address}: {e}")
            return None


class TweetScoutAPI:
    """API Interface for TweetScout.io"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.tweetscout.io/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "TweetScout_Analysis_Bot/1.0",
            "Accept": "application/json"
        }
    
    def analyze_account(self, twitter_handle):
        """Analyze a Twitter account"""
        endpoint = f"{self.base_url}/account/analyze"
        params = {"handle": twitter_handle}
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to analyze Twitter account {twitter_handle}: {e}")
            return None
    
    def search_mentions(self, contract_address, days_back=7):
        """Search for mentions of a contract address"""
        endpoint = f"{self.base_url}/search"
        params = {
            "query": contract_address,
            "days": days_back
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to search mentions for {contract_address}: {e}")
            return None


class SecurityScoreAPI:
    """API Interface for security scoring platforms"""
    def __init__(self, solana_sniffer_key, rugcheck_key):
        self.solana_sniffer_key = solana_sniffer_key
        self.rugcheck_key = rugcheck_key
        self.solana_sniffer_url = "https://api.solanasniffer.com/v1"
        self.rugcheck_url = "https://api.rugcheck.xyz/v1"
    
    def get_solana_sniffer_score(self, contract_address):
        """Get security score from SolanaSniffer"""
        endpoint = f"{self.solana_sniffer_url}/token/{contract_address}/score"
        headers = {
            "X-API-Key": self.solana_sniffer_key,
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get SolanaSniffer score for {contract_address}: {e}")
            return None
    
    def get_rugcheck_score(self, contract_address):
        """Get security score from RugCheckXYZ"""
        endpoint = f"{self.rugcheck_url}/audit/{contract_address}"
        headers = {
            "Authorization": f"Bearer {self.rugcheck_key}",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get RugCheck score for {contract_address}: {e}")
            return None
    
    def get_combined_security_score(self, contract_address):
        """Calculate combined security score from multiple sources"""
        solana_score = self.get_solana_sniffer_score(contract_address)
        rugcheck_score = self.get_rugcheck_score(contract_address)
        
        scores = []
        
        if solana_score and 'score' in solana_score:
            scores.append(solana_score['score'])
        
        if rugcheck_score and 'score' in rugcheck_score:
            scores.append(rugcheck_score['score'])
        
        if not scores:
            logger.warning(f"No security scores available for {contract_address}")
            return None
        
        combined_score = sum(scores) / len(scores)
        logger.info(f"Combined security score for {contract_address}: {combined_score}")
        
        return {
            'combined_score': combined_score,
            'solana_score': solana_score.get('score') if solana_score else None,
            'rugcheck_score': rugcheck_score.get('score') if rugcheck_score else None,
            'issues': solana_score.get('issues', []) if solana_score else []
        }


class AIAnalyzer:
    """AI-powered token analysis and prediction"""
    def __init__(self, token_db):
        self.token_db = token_db
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_training_data(self):
        """Prepare historical data for model training"""
        if not self.token_db.tokens:
            logger.warning("No tokens available for training AI model")
            return None, None
        
        data = []
        labels = []
        
        for addr, token in self.token_db.tokens.items():
            # Skip tokens without complete data
            if not all(k in token for k in ['social_engagement', 'holder_count', 
                                          'liquidity', 'initial_price', 'peak_price']):
                continue
            
            # Feature extraction
            features = [
                token.get('social_engagement', 0),
                token.get('holder_count', 0),
                token.get('liquidity', 0),
                token.get('developer_previous_projects', 0),
                token.get('security_score', 50)
            ]
            
            # Success label (1 if 3x or more, 0 otherwise)
            success = 1 if token.get('peak_price', 0) / token.get('initial_price', 1) >= 3 else 0
            
            data.append(features)
            labels.append(success)
        
        if not data:
            logger.warning("Insufficient data for AI training")
            return None, None
        
        return np.array(data), np.array(labels)
    
    def train_model(self):
        """Train the AI model for token success prediction"""
        X, y = self.prepare_training_data()
        
        if X is None or len(X) < 10:
            logger.warning("Not enough data to train a reliable model")
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train RandomForest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Calculate and log accuracy
        accuracy = self.model.score(X_scaled, y)
        logger.info(f"AI model trained with accuracy: {accuracy:.2f}")
        
        return True
    
    def predict_success_probability(self, token_data):
        """Predict success probability for a token"""
        if self.model is None:
            if not self.train_model():
                logger.warning("Unable to make prediction - model not trained")
                return None
        
        # Extract features
        features = np.array([
            token_data.get('social_engagement', 0),
            token_data.get('holder_count', 0),
            token_data.get('liquidity', 0),
            token_data.get('developer_previous_projects', 0),
            token_data.get('security_score', 50)
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probability of success
        success_prob = self.model.predict_proba(features_scaled)[0][1]
        
        return success_prob
    
    def analyze_token_comprehensively(self, token_address, token_data, social_data, security_data):
        """Perform comprehensive analysis on a token"""
        analysis_result = {
            'token_address': token_address,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Basic metrics
        if token_data:
            analysis_result['metrics'].update({
                'name': token_data.get('name', 'Unknown'),
                'symbol': token_data.get('symbol', 'Unknown'),
                'current_price': token_data.get('price', 0),
                'market_cap': token_data.get('market_cap', 0),
                'holder_count': token_data.get('holders', 0)
            })
        
        # Social metrics
        if social_data:
            analysis_result['metrics'].update({
                'twitter_followers': social_data.get('twitter_followers', 0),
                'twitter_engagement': social_data.get('engagement_rate', 0),
                'sentiment_score': social_data.get('sentiment', 0),
                'mentions_24h': social_data.get('mentions_24h', 0)
            })
        
        # Security metrics
        if security_data:
            analysis_result['metrics'].update({
                'security_score': security_data.get('combined_score', 0),
                'risk_level': 'High' if security_data.get('combined_score', 0) < 60 else
                              'Medium' if security_data.get('combined_score', 0) < 80 else 'Low',
                'security_issues': len(security_data.get('issues', []))
            })
        
        # Predict success probability
        combined_data = {**analysis_result['metrics']}
        if token_data:
            combined_data.update({
                'social_engagement': social_data.get('engagement_rate', 0) if social_data else 0,
                'security_score': security_data.get('combined_score', 50) if security_data else 50,
                'liquidity': token_data.get('liquidity', 0),
                'developer_previous_projects': token_data.get('developer_project_count', 0)
            })
        
        success_prob = self.predict_success_probability(combined_data)
        if success_prob is not None:
            analysis_result['success_probability'] = success_prob
            analysis_result['recommendation'] = 'Strong Buy' if success_prob > 0.8 else 'Buy' if success_prob > 0.6 else 'Hold' if success_prob > 0.4 else 'Avoid'
        
        return analysis_result


class SolanaTrader:
    """Solana token trading implementation"""
    def __init__(self, config, client=None):
        self.config = config
        self.client = client or Client(config.config['solana_rpc_url'])
        
        # Initialize wallet if private key is provided
        if config.config['wallet_private_key']:
            try:
                self.keypair = Keypair.from_secret_key(
                    bytes.fromhex(config.config['wallet_private_key'])
                )
                self.public_key = self.keypair.public_key
                logger.info(f"Wallet initialized with public key: {self.public_key}")
            except Exception as e:
                logger.error(f"Failed to initialize wallet: {e}")
                self.keypair = None
                self.public_key = None
    
    def check_wallet_balance(self):
        """Check SOL balance in the wallet"""
        if not self.public_key:
            logger.error("No wallet initialized")
            return None
        
        try:
            response = self.client.get_balance(self.public_key)
            balance_lamports = response['result']['value']
            balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
            logger.info(f"Wallet balance: {balance_sol} SOL")
            return balance_sol
        except Exception as e:
            logger.error(f"Failed to check wallet balance: {e}")
            return None
    
    def buy_token(self, token_address, amount_sol):
        """Buy a token with specified SOL amount"""
        if not self.keypair:
            logger.error("No wallet initialized")
            return False
        
        logger.info(f"Attempting to buy {token_address} for {amount_sol} SOL")
        
        # Implementation would depend on specific DEX integration
        # This is a placeholder for the actual implementation
        try:
            # Create transaction with priority fees
            priority_fee_lamports = int(self.config.config['priority_fee'] * 1_000_000_000)
            slippage = self.config.config['buy_slippage'] / 100
            
            # Mock transaction structure - actual implementation would use Jupiter or similar
            # Create and sign transaction
            # Submit transaction to blockchain
            
            logger.info(f"Buy order submitted for {token_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to buy token {token_address}: {e}")
            return False
    
    def sell_token(self, token_address, percentage=100):
        """Sell specified percentage of token holdings"""
        if not self.keypair:
            logger.error("No wallet initialized")
            return False
        
        logger.info(f"Attempting to sell {percentage}% of {token_address}")
        
        # Implementation would depend on specific DEX integration
        try:
            # Create transaction with priority fees
            priority_fee_lamports = int(self.config.config['priority_fee'] * 1_000_000_000)
            slippage = self.config.config['sell_slippage'] / 100
            
            # Mock transaction structure - actual implementation would use Jupiter or similar
            # Create and sign transaction
            # Submit transaction to blockchain
            
            logger.info(f"Sell order submitted for {percentage}% of {token_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to sell token {token_address}: {e}")
            return False


class TwitterMonitor:
    """Monitor Twitter for KOL mentions of contract addresses"""
    def __init__(self, config, api_keys=None):
        self.config = config
        
        # Set up Twitter API client if credentials provided
        if api_keys and all(k in api_keys for k in 
                           ['consumer_key', 'consumer_secret', 'access_token', 'access_secret']):
            try:
                auth = tweepy.OAuthHandler(api_keys['consumer_key'], api_keys['consumer_secret'])
                auth.set_access_token(api_keys['access_token'], api_keys['access_secret'])
                self.client = tweepy.API(auth)
                logger.info("Twitter API client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
                self.client = None
        else:
            logger.warning("No Twitter API credentials provided")
            self.client = None
    
    def extract_contract_addresses(self, tweet_text):
        """Extract potential Solana contract addresses from tweet text"""
        # Simplified regex pattern for Solana addresses
        import re
        pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
        matches = re.findall(pattern, tweet_text)
        return matches
    
    def check_kol_tweets(self, accounts_to_check):
        """Check recent tweets from KOLs for contract addresses"""
        if not self.client:
            logger.error("Twitter client not initialized")
            return []
        
        found_addresses = []
        
        for account in accounts_to_check:
            try:
                tweets = self.client.user_timeline(screen_name=account, count=20, tweet_mode="extended")
                logger.info(f"Retrieved {len(tweets)} tweets from {account}")
                
                for tweet in tweets:
                    addresses = self.extract_contract_addresses(tweet.full_text)
                    if addresses:
                        logger.info(f"Found contract addresses in tweet from {account}: {addresses}")
                        for addr in addresses:
                            found_addresses.append({
                                'address': addr,
                                'kol': account,
                                'tweet_id': tweet.id,
                                'tweet_text': tweet.full_text,
                                'timestamp': tweet.created_at.isoformat()
                            })
            except Exception as e:
                logger.error(f"Error checking tweets from {account}: {e}")
        
        return found_addresses


class PumpFunBotRunner:
    """Main bot runner class that orchestrates all components"""
    def __init__(self, config_path='config.json'):
        self.config = Config(config_path)
        self.token_db = TokenDatabase()
        self.pumpfun_api = PumpFunAPI()
        self.gmgn_api = GMGNMonitor()
        
        # Initialize APIs with keys from config
        self.tweetscout = TweetScoutAPI(self.config.config['tweetscout_api_key']) if self.config.config['tweetscout_api_key'] else None
        self.security_api = SecurityScoreAPI(
            self.config.config['solanasniffer_api_key'],
            self.config.config['rugcheck_api_key']
        ) if self.config.config['solanasniffer_api_key'] and self.config.config['rugcheck_api_key'] else None
        
        # Initialize AI analyzer
        self.ai_analyzer = AIAnalyzer(self.token_db)
        
        # Initialize trader
        self.trader = SolanaTrader(self.config)
        
        # Initialize Twitter monitor
        self.twitter_monitor = TwitterMonitor(self.config)
        
        logger.info("PumpFun Bot initialized successfully")
    
    def scan_migrated_tokens(self, limit=100):
        """Scan and analyze migrated tokens"""
        logger.info(f"Scanning up to {limit} migrated tokens")
        
        migrated_tokens = self.pumpfun_api.get_migrated_tokens(limit=limit)
        if not migrated_tokens:
            logger.warning("Failed to retrieve migrated tokens")
            return
        
        for token in migrated_tokens:
            token_address = token.get('address')
            
            # Skip blacklisted coins
            if token_address in self.config.config['blacklisted_coins']:
                logger.info(f"Skipping blacklisted token: {token_address}")
                continue
            
            # Skip blacklisted developers
            developer = token.get('developer')
            if developer in self.config.config['blacklisted_devs']:
                logger.info(f"Skipping token from blacklisted developer: {developer}")
                continue
            
            # Get detailed token information
            token_details = self.pumpfun_api.get_token_details(token_address)
            if not token_details:
                logger.warning(f"Failed to retrieve details for token {token_address}")
                continue
            
            # Get security score
            security_score = None
            if self.security_api:
                security_score = self.security_api.get_combined_security_score(token_address)
                if security_score and security_score['combined_score'] < self.config.config['minimum_token_score']:
                    logger.warning(f"Token {token_address} has a low security score: {security_score['combined_score']}")
                    continue
            
            # Get social data
            social_data = None
            if self.tweetscout:
                if token_details.get('twitter'):
                    social_data = self.tweetscout.analyze_account(token_details['twitter'])
            
            # Analyze token
            # Analyze token
            token_analysis = self.ai_analyzer.analyze_token_comprehensively(
                token_address, token_details, social_data, security_score
            )
            
            # Store analysis results
            self.token_db.add_token(token_address, {
                **token_details,
                'migration_data': token.get('migration_data'),
                'migration_timestamp': token.get('migration_timestamp'),
                'security_score': security_score['combined_score'] if security_score else None,
                'analysis': token_analysis
            })
            
            logger.info(f"Analyzed and stored token {token_address}")
        
        # Find patterns across tokens
        patterns = self.token_db.find_migration_patterns()
        logger.info(f"Found {len(patterns)} migration patterns")
        return patterns
    
    def monitor_gmgn_tokens(self, limit=20):
        """Monitor trending tokens on GMGN"""
        logger.info("Monitoring GMGN trending tokens")
        
        trending = self.gmgn_api.get_trending_tokens(limit=limit)
        if not trending:
            logger.warning("Failed to retrieve trending tokens from GMGN")
            return []
        
        analyzed_tokens = []
        
        for token in trending:
            token_address = token.get('address')
            
            # Skip blacklisted coins/devs
            if (token_address in self.config.config['blacklisted_coins'] or
                token.get('developer') in self.config.config['blacklisted_devs']):
                continue
            
            # Get security score
            security_score = None
            if self.security_api:
                security_score = self.security_api.get_combined_security_score(token_address)
                if security_score and security_score['combined_score'] < self.config.config['minimum_token_score']:
                    logger.warning(f"Token {token_address} has a low security score: {security_score['combined_score']}")
                    continue
            
            # Get social data
            social_data = self.gmgn_api.get_token_social_data(token_address)
            
            # Analyze token
            token_analysis = self.ai_analyzer.analyze_token_comprehensively(
                token_address, token, social_data, security_score
            )
            
            # Store token data
            self.token_db.add_token(token_address, {
                **token,
                'security_score': security_score['combined_score'] if security_score else None,
                'analysis': token_analysis,
                'source': 'gmgn_trending'
            })
            
            analyzed_tokens.append({
                'address': token_address,
                'symbol': token.get('symbol'),
                'success_probability': token_analysis.get('success_probability'),
                'recommendation': token_analysis.get('recommendation')
            })
            
        logger.info(f"Analyzed {len(analyzed_tokens)} trending tokens from GMGN")
        return analyzed_tokens
    
    def compare_token_sources(self):
        """Compare tokens from different sources to find patterns"""
        migrated_tokens = {addr: data for addr, data in self.token_db.tokens.items() 
                          if data.get('migration_data')}
        gmgn_tokens = {addr: data for addr, data in self.token_db.tokens.items() 
                      if data.get('source') == 'gmgn_trending'}
        
        if not migrated_tokens or not gmgn_tokens:
            logger.warning("Insufficient data to compare token sources")
            return None
        
        # Convert to dataframes for analysis
        df_migrated = pd.DataFrame.from_dict(migrated_tokens, orient='index')
        df_gmgn = pd.DataFrame.from_dict(gmgn_tokens, orient='index')
        
        # Prepare comparison result
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'sample_sizes': {
                'migrated': len(df_migrated),
                'gmgn_trending': len(df_gmgn)
            },
            'success_rates': {},
            'common_patterns': [],
            'differences': []
        }
        
        # Compare success rates if prediction data is available
        if 'analysis' in df_migrated.columns and 'analysis' in df_gmgn.columns:
            migrated_success = df_migrated['analysis'].apply(
                lambda x: x.get('success_probability') if isinstance(x, dict) else None
            ).dropna().mean()
            
            gmgn_success = df_gmgn['analysis'].apply(
                lambda x: x.get('success_probability') if isinstance(x, dict) else None
            ).dropna().mean()
            
            comparison['success_rates'] = {
                'migrated_tokens': migrated_success,
                'gmgn_trending': gmgn_success
            }
        
        # Find common successful patterns
        # This is simplified - a real implementation would do deeper analysis
        common_patterns = []
        
        # Check for developer overlap
        if 'developer' in df_migrated.columns and 'developer' in df_gmgn.columns:
            migrated_devs = set(df_migrated['developer'].dropna())
            gmgn_devs = set(df_gmgn['developer'].dropna())
            common_devs = migrated_devs.intersection(gmgn_devs)
            
            if common_devs:
                common_patterns.append({
                    'type': 'developer_overlap',
                    'description': f"Found {len(common_devs)} developers with tokens in both categories",
                    'developers': list(common_devs)
                })
        
        # Check for social media patterns
        # Simplified example - a real implementation would be more sophisticated
        comparison['common_patterns'] = common_patterns
        
        logger.info(f"Completed comparison between migrated and trending tokens")
        return comparison
    
    def monitor_twitter_kols(self):
        """Monitor Twitter KOLs for token mentions"""
        if not self.twitter_monitor or not self.config.config['twitter_accounts_to_monitor']:
            logger.warning("Twitter monitoring not configured")
            return []
        
        logger.info(f"Monitoring {len(self.config.config['twitter_accounts_to_monitor'])} Twitter KOLs")
        found_addresses = self.twitter_monitor.check_kol_tweets(
            self.config.config['twitter_accounts_to_monitor']
        )
        
        if not found_addresses:
            logger.info("No contract addresses found in recent KOL tweets")
            return []
        
        analyzed_addresses = []
        
        for item in found_addresses:
            token_address = item['address']
            
            # Skip blacklisted coins
            if token_address in self.config.config['blacklisted_coins']:
                continue
            
            # Get token details (attempt PumpFun first, then GMGN)
            token_details = self.pumpfun_api.get_token_details(token_address)
            if not token_details:
                logger.info(f"Token {token_address} not found on PumpFun, checking GMGN...")
                # Implement fallback to GMGN API
            
            # Get security score
            security_score = None
            if self.security_api:
                security_score = self.security_api.get_combined_security_score(token_address)
                if security_score and security_score['combined_score'] < self.config.config['minimum_token_score']:
                    logger.warning(f"Token {token_address} mentioned by KOL has low security score: {security_score['combined_score']}")
                    continue
            
            # Store the mention and analysis
            if token_details:
                self.token_db.add_token(token_address, {
                    **token_details,
                    'kol_mention': item,
                    'security_score': security_score['combined_score'] if security_score else None,
                    'source': 'twitter_kol'
                })
                
                analyzed_addresses.append({
                    'address': token_address,
                    'symbol': token_details.get('symbol', 'Unknown'),
                    'kol': item['kol'],
                    'security_score': security_score['combined_score'] if security_score else None
                })
        
        logger.info(f"Found and analyzed {len(analyzed_addresses)} tokens from KOL tweets")
        return analyzed_addresses
    
    def execute_trading_strategy(self, token_address, investment_amount=1):
        """Execute trading strategy for a specific token"""
        if not self.trader or not self.trader.keypair:
            logger.error("Trader not initialized with wallet")
            return False
        
        # Check wallet balance
        balance = self.trader.check_wallet_balance()
        if not balance or balance < investment_amount:
            logger.error(f"Insufficient balance for trading: {balance} SOL")
            return False
        
        # Get token analysis
        token = self.token_db.get_token(token_address)
        if not token:
            logger.error(f"No analysis data for token {token_address}")
            return False
        
        analysis = token.get('analysis', {})
        success_prob = analysis.get('success_probability')
        
        # Trading logic
        if success_prob and success_prob > 0.7:
            logger.info(f"High success probability ({success_prob:.2f}) for {token_address}, executing buy")
            
            # Execute buy
            buy_success = self.trader.buy_token(token_address, investment_amount)
            if not buy_success:
                logger.error(f"Failed to buy token {token_address}")
                return False
            
            # Set up monitoring for profit targets
            profit_target = self.config.config['profit_target']
            long_term_percentage = self.config.config['long_term_percentage']
            
            logger.info(f"Buy executed for {token_address}. Target: {profit_target}x, holding {long_term_percentage}% long-term")
            return True
        else:
            logger.info(f"Token {token_address} doesn't meet trading criteria (success prob: {success_prob})")
            return False
    
    def run_scheduled_tasks(self):
        """Run all scheduled monitoring and analysis tasks"""
        # 1. Scan migrated tokens
        migrated_patterns = self.scan_migrated_tokens(limit=50)
        
        # 2. Monitor GMGN trending tokens
        gmgn_trending = self.monitor_gmgn_tokens(limit=20)
        
        # 3. Compare data sources
        comparison = self.compare_token_sources()
        
        # 4. Monitor Twitter KOLs
        kol_mentions = self.monitor_twitter_kols()
        
        # 5. Run model training periodically
        self.ai_analyzer.train_model()
        
        # 6. Find trading opportunities
        trading_opportunities = []
        
        # Prioritize KOL mentions with good security scores
        for mention in kol_mentions:
            if mention.get('security_score') and mention['security_score'] > 75:
                trading_opportunities.append({
                    'address': mention['address'],
                    'source': 'kol_mention',
                    'priority': 'high'
                })
        
        # Add top GMGN trending tokens
        for token in sorted(gmgn_trending, 
                           key=lambda x: x.get('success_probability', 0), 
                           reverse=True)[:3]:
            if token.get('success_probability', 0) > 0.7:
                trading_opportunities.append({
                    'address': token['address'],
                    'source': 'gmgn_trending',
                    'priority': 'medium'
                })
        
        # Execute trades for opportunities
        executed_trades = 0
        for opportunity in trading_opportunities:
            if executed_trades >= 3:  # Limit number of trades per run
                break
                
            success = self.execute_trading_strategy(
                opportunity['address'],
                investment_amount=2 if opportunity['priority'] == 'high' else 1
            )
            
            if success:
                executed_trades += 1
        
        logger.info(f"Scheduled tasks completed. Executed {executed_trades} trades.")
        return {
            'migrated_patterns': len(migrated_patterns) if migrated_patterns else 0,
            'gmgn_trending_analyzed': len(gmgn_trending),
            'kol_mentions': len(kol_mentions),
            'trades_executed': executed_trades
        }


def main():
    """Main entry point for the bot"""
    print("Starting PumpFun and GMGN Trading Bot...")
    
    # Initialize the bot
    bot = PumpFunBotRunner()
    
    # Run once immediately
    results = bot.run_scheduled_tasks()
    print(f"Initial run completed: {results}")
    
    # Enter monitoring loop
    try:
        while True:
            print("Waiting for next scheduled run...")
            time.sleep(3600)  # Run hourly
            results = bot.run_scheduled_tasks()
            print(f"Scheduled run completed: {results}")
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        logger.error(f"Critical error in main loop: {e}", exc_info=True)
    finally:
        print("Saving database before exit...")
        bot.token_db.save_database()
        print("Bot shutdown complete")


if __name__ == "__main__":
    main()