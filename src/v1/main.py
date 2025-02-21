#!/usr/bin/python3
# Author: @blankgodd || @AgbaD

import os
from dotenv import load_dotenv
from web3 import Web3, HTTPProvider
from collections import defaultdict

load_dotenv()


class Pylyser:
    def __init__(self, rpc_url) -> None:
        self.w3 = Web3(HTTPProvider(rpc_url))

    def get_latest_block_number(self):
        return self.w3.eth.block_number
    
    def get_block_min(self, block_number):
        return self.w3.eth.get_block(block_number, False)
    
    def get_block_full(self, block_number):
        return self.w3.eth.get_block(block_number, True)
    
    def get_transaction_by_hash(self, txn_hash):
        return self.w3.eth.get_transaction(txn_hash)
    
    def convert_wei_to_ether(self, wei_amount):
        return Web3.from_wei(wei_amount, 'ether')
    
    def analyze_transaction(self, txn):
        tx_balances = defaultdict(lambda: 0)

        tx_value = self.convert_wei_to_ether(txn['value'])
        tx_balances[txn['from']] -= tx_value
        tx_balances[txn['to']] += tx_value
        tx_volume = tx_value
        
        return tx_balances, tx_volume

    
    def analyze_block(self, block):
        block_balances = defaultdict(lambda: 0)
        block_volume = 0

        for txn in block.transactions[1:]:
            tx = self.get_transaction_by_hash(txn)
            tx_balances, tx_volume = self.analyze_transaction(tx)
            block_volume += tx_volume

            for addr, balance in tx_balances.items():
                block_balances[addr] += balance
        return block_balances, block_volume
    

    def analyze_blockchain(self, start, end):
        blockchain_balances = defaultdict(lambda: 0)
        blockwise_metrics = {}

        while start <= end:
            block = self.get_block_min(start)
            block_balances, block_volume = self.analyze_block(block)

            for addr, balance in block_balances.items():
                blockchain_balances[addr] += balance

            metrics = {
                "number_of_addresses": len(blockchain_balances.keys()),
                "transactions_volume": block_volume,
                "utxos": sum(blockchain_balances.values())
            }
            blockwise_metrics[start] = metrics
            start += 1

        return blockchain_balances, blockwise_metrics


if __name__ == "__main__":
    lyser = Pylyser(os.getenv('RPC_URL'))
    latest_block = lyser.get_latest_block_number()
    resp = lyser.analyze_blockchain(latest_block-3, latest_block)

