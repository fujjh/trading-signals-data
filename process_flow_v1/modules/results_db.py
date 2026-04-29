#!/usr/bin/env python3
"""
================================================================================
RESULTS DATABASE MODULE
================================================================================

SQLite database for storing backtest results, metrics, and optimization history.
Enables query-based analysis and A/B comparison of strategies.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-28
================================================================================
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from metrics_calculator import PerformanceMetrics


@dataclass
class BacktestResult:
    """Single backtest result record"""
    id: Optional[int]
    run_date: str
    config_name: str
    config_json: str
    
    # Parameters
    initial_capital: float
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    hold_days: int
    
    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit metrics
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    expectancy: float
    avg_trade: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float
    
    # Additional data
    trades_json: str
    metrics_json: str
    notes: str


class ResultsDatabase:
    """SQLite database manager for backtest results"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database (default: data/results.db)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "results.db"
        
        self.db_path = db_path
        self.conn = None
        self._ensure_db()
    
    def _ensure_db(self):
        """Create database and tables if not exists"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create required tables"""
        cursor = self.conn.cursor()
        
        # Main backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                config_name TEXT NOT NULL,
                config_json TEXT,
                
                initial_capital REAL,
                position_size_pct REAL,
                stop_loss_pct REAL,
                take_profit_pct REAL,
                hold_days INTEGER,
                
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                
                gross_profit REAL,
                gross_loss REAL,
                net_profit REAL,
                profit_factor REAL,
                expectancy REAL,
                avg_trade REAL,
                
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                calmar_ratio REAL,
                volatility REAL,
                
                trades_json TEXT,
                metrics_json TEXT,
                notes TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Optimization runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                population_size INTEGER,
                generations INTEGER,
                best_config_json TEXT,
                best_fitness REAL,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Walk-forward results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS walk_forward_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimization_id INTEGER,
                window_number INTEGER,
                train_start TEXT,
                train_end TEXT,
                test_start TEXT,
                test_end TEXT,
                train_metrics_json TEXT,
                test_metrics_json TEXT,
                is_consistent INTEGER,
                
                FOREIGN KEY (optimization_id) REFERENCES optimization_runs(id)
            )
        ''')
        
        # Configuration versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                parent_version_id INTEGER,
                description TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_version_id) REFERENCES config_versions(id)
            )
        ''')
        
        self.conn.commit()
    
    def save_backtest_result(self, 
                            config_name: str,
                            config: Dict[str, Any],
                            metrics: PerformanceMetrics,
                            trades: List[Dict],
                            notes: str = "") -> int:
        """
        Save a backtest result to database
        
        Returns:
            ID of inserted record
        """
        result = BacktestResult(
            id=None,
            run_date=datetime.now().isoformat(),
            config_name=config_name,
            config_json=json.dumps(config),
            
            initial_capital=config.get('initial_capital', 100000),
            position_size_pct=config.get('position_size_pct', 0.05),
            stop_loss_pct=config.get('stop_loss_pct', 0.05),
            take_profit_pct=config.get('take_profit_pct', 0.10),
            hold_days=config.get('hold_days', 5),
            
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            win_rate=metrics.win_rate,
            
            gross_profit=metrics.gross_profit,
            gross_loss=metrics.gross_loss,
            net_profit=metrics.net_profit,
            profit_factor=metrics.profit_factor,
            expectancy=metrics.expectancy,
            avg_trade=metrics.avg_trade,
            
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.max_drawdown,
            calmar_ratio=metrics.calmar_ratio,
            volatility=metrics.volatility,
            
            trades_json=json.dumps(trades),
            metrics_json=json.dumps(self._metrics_to_dict(metrics)),
            notes=notes
        )
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO backtest_results (
                run_date, config_name, config_json,
                initial_capital, position_size_pct, stop_loss_pct, take_profit_pct, hold_days,
                total_trades, winning_trades, losing_trades, win_rate,
                gross_profit, gross_loss, net_profit, profit_factor, expectancy, avg_trade,
                sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, volatility,
                trades_json, metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.run_date, result.config_name, result.config_json,
            result.initial_capital, result.position_size_pct, result.stop_loss_pct, 
            result.take_profit_pct, result.hold_days,
            result.total_trades, result.winning_trades, result.losing_trades, result.win_rate,
            result.gross_profit, result.gross_loss, result.net_profit, result.profit_factor,
            result.expectancy, result.avg_trade,
            result.sharpe_ratio, result.sortino_ratio, result.max_drawdown, result.calmar_ratio,
            result.volatility,
            result.trades_json, result.metrics_json, result.notes
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_backtest_result(self, result_id: int) -> Optional[Dict]:
        """Retrieve a backtest result by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM backtest_results WHERE id = ?', (result_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(row, cursor.description)
        return None
    
    def get_all_results(self, config_name: Optional[str] = None) -> pd.DataFrame:
        """Get all results, optionally filtered by config name"""
        query = 'SELECT * FROM backtest_results'
        params = []
        
        if config_name:
            query += ' WHERE config_name = ?'
            params.append(config_name)
        
        query += ' ORDER BY run_date DESC'
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def compare_configs(self, config_names: List[str]) -> pd.DataFrame:
        """Compare performance of different configurations"""
        placeholders = ','.join('?' * len(config_names))
        query = f'''
            SELECT config_name,
                   AVG(profit_factor) as avg_profit_factor,
                   AVG(sharpe_ratio) as avg_sharpe,
                   AVG(max_drawdown) as avg_max_dd,
                   AVG(win_rate) as avg_win_rate,
                   AVG(net_profit) as avg_net_profit,
                   COUNT(*) as run_count
            FROM backtest_results
            WHERE config_name IN ({placeholders})
            GROUP BY config_name
        '''
        
        return pd.read_sql_query(query, self.conn, params=config_names)
    
    def get_best_result(self, metric: str = 'profit_factor') -> Optional[Dict]:
        """Get the best performing result by metric"""
        valid_metrics = [
            'profit_factor', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'net_profit', 'win_rate', 'expectancy'
        ]
        
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")
        
        # For drawdown, lower is better
        order = 'ASC' if metric == 'max_drawdown' else 'DESC'
        
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT * FROM backtest_results 
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} {order}
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row, cursor.description)
        return None
    
    def save_optimization_run(self, 
                             run_name: str,
                             population_size: int,
                             generations: int,
                             best_config: Dict,
                             best_fitness: float) -> int:
        """Save an optimization run"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO optimization_runs (run_name, start_date, population_size, generations, best_config_json, best_fitness)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            run_name, datetime.now().isoformat(), population_size, generations,
            json.dumps(best_config), best_fitness
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def save_config_version(self, 
                           version_name: str,
                           config: Dict,
                           parent_version_id: Optional[int] = None,
                           description: str = "") -> int:
        """Save a configuration version"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO config_versions (version_name, config_json, parent_version_id, description)
            VALUES (?, ?, ?, ?)
        ''', (version_name, json.dumps(config), parent_version_id, description))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_config_version(self, version_id: int) -> Optional[Dict]:
        """Retrieve a configuration version"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM config_versions WHERE id = ?', (version_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'version_name': row[1],
                'config': json.loads(row[2]),
                'parent_version_id': row[3],
                'description': row[4],
                'created_at': row[5]
            }
        return None
    
    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict:
        """Convert PerformanceMetrics to dictionary"""
        return {
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'gross_profit': metrics.gross_profit,
            'gross_loss': metrics.gross_loss,
            'net_profit': metrics.net_profit,
            'profit_factor': metrics.profit_factor,
            'expectancy': metrics.expectancy,
            'payoff_ratio': metrics.payoff_ratio,
            'avg_trade': metrics.avg_trade,
            'avg_win': metrics.avg_win,
            'avg_loss': metrics.avg_loss,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_duration': metrics.max_drawdown_duration,
            'avg_drawdown': metrics.avg_drawdown,
            'ulcer_index': metrics.ulcer_index,
            'volatility': metrics.volatility,
            'downside_volatility': metrics.downside_volatility,
            'max_consecutive_wins': metrics.max_consecutive_wins,
            'max_consecutive_losses': metrics.max_consecutive_losses
        }
    
    def _row_to_dict(self, row: sqlite3.Row, description) -> Dict:
        """Convert database row to dictionary"""
        return {desc[0]: value for desc, value in zip(description, row)}
    
    def export_to_csv(self, output_path: Path):
        """Export all results to CSV"""
        df = self.get_all_results()
        df.to_csv(output_path, index=False)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def demo():
    """Demonstrate database functionality"""
    with ResultsDatabase() as db:
        # Create sample metrics
        from metrics_calculator import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.60,
            gross_profit=15000,
            gross_loss=5000,
            net_profit=10000,
            profit_factor=3.0,
            expectancy=100,
            payoff_ratio=1.5,
            avg_trade=100,
            avg_win=250,
            avg_loss=125,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=0.15,
            max_drawdown_duration=20,
            avg_drawdown=0.05,
            ulcer_index=0.03,
            total_return=0.20,
            annualized_return=0.25,
            volatility=0.15,
            downside_volatility=0.10,
            consecutive_wins=5,
            consecutive_losses=3,
            max_consecutive_wins=8,
            max_consecutive_losses=5
        )
        
        config = {
            'initial_capital': 100000,
            'position_size_pct': 0.20,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'hold_days': 5
        }
        
        trades = [
            {'date': '2026-01-01', 'pnl': 500},
            {'date': '2026-01-02', 'pnl': -200}
        ]
        
        # Save result
        result_id = db.save_backtest_result(
            config_name='test_config_v1',
            config=config,
            metrics=metrics,
            trades=trades,
            notes='Test run'
        )
        
        print(f"Saved result ID: {result_id}")
        
        # Retrieve result
        result = db.get_backtest_result(result_id)
        print(f"\nRetrieved result:")
        print(f"  Config: {result['config_name']}")
        print(f"  Profit Factor: {result['profit_factor']}")
        print(f"  Sharpe: {result['sharpe_ratio']}")
        
        # Get best result
        best = db.get_best_result('profit_factor')
        if best:
            print(f"\nBest by Profit Factor: {best['config_name']}")


if __name__ == "__main__":
    demo()
