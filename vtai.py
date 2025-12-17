# Dependencies Imports
import os

# Local imports
from utils.basics import load_config, welcome_message
from src.data.data_wrangling import data_wrangling
from src.data.feature_enginiering import feature_enginiering
from src.data.data_split import data_split
from src.data.targeting import targerting

# Local Constants
CONFIG_PATH = "config/config.yaml"
REPORTS_PATH = "reports/"

def main():
    # Load Config
    config = load_config(CONFIG_PATH)

    # Config Dictionaries
    basic_config = {
        'name': config.get('name', 'VTAI'),
        'version': config.get('version', ''),
        'reports_name': config.get('reports_name', 'Default Report'),
        'config_path': CONFIG_PATH,
        'reports_path': REPORTS_PATH
    }

    data_wrangling_config = {
        'data_path': config.get('data_path', 'sample_data/raw_data/XAUUSD.csv'),
        'output_path': os.path.join('sample_data', basic_config['reports_name'], 'cleaned_data'),
        'tick_size': config.get('tick_bar_size', 200)
    }

    feature_enginiering_config = {
        'data_soruce_path': data_wrangling_config.get('output_path', os.path.join('sample_data', basic_config['reports_name'], 'cleaned_data.parquet')),
        'output_path': os.path.join('sample_data', basic_config['reports_name'], 'featured_data')
    }

    data_split_config = {
        'features_source_path': f'{feature_enginiering_config.get('output_path', os.path.join('sample_data', basic_config['reports_name'], 'featured_data'))}.parquet',
        'output_path': os.path.join('sample_data', basic_config['reports_name'], 'splits'),
        'starting_date': config.get('starting_date', '2025-01-01'),
        'end_date': config.get('end_date', '2025-12-01'),
        'train_poriton': config.get('train_portion', 0.7),
        'val_portion': config.get('val_portion', 0.15),
        'horizon_bars': config.get('horizon_bars', 50),
        'vol_windows': config.get('vol_windows', 50),
        'vol_mode': config.get('vol_mode', 'ewm'),
        'vol_ret_clip': config.get('vol_ret_clip', 0.02),
        'pt_mult': config.get('pt_mult', 1.0),
        'sl_mult': config.get('sl_mult', 1.0),
        'cost_k_spread': config.get('cost_k_spread', 0.5),
        'slippage_k': config.get('slippage_k', 0.25),
        'slippage_pow_speed': config.get('slippage_pow_speed', 0.35),
        'min_cost_bps': config.get('min_cost_bps', 0.25),
        'max_cost_bps': config.get('max_cost_bps', 8),
        'cost_roundtrip': config.get('cost_roundtrip', True),
        'min_vol': config.get('min_vol', 1e-6),
        'ret_clip': config.get('ret_clip', 0.2),
        'tie_break': config.get('tie_break', 'close_based'),
        'purge_mode': config.get('purge_mode', 'none'),
        'purge_embargo_bars': config.get('purge_embargo_bars', 0),
        'weight_cap': config.get('weight_cap', 50)
    }
    
    targeting_config = {
        'train_source_path': f'{os.path.join(data_split_config.get('output_path', f'sample_data/{basic_config['reports_name']}/splits'), 'train.parquet')}',
        'val_source_path': f'{os.path.join(data_split_config.get('output_path', f'sample_data/{basic_config['reports_name']}/splits'), 'val.parquet')}',
        'test_source_path': f'{os.path.join(data_split_config.get('output_path', f'sample_data/{basic_config['reports_name']}/splits'), 'test.parquet')}',
        'output_path': os.path.join('sample_data', basic_config['reports_name'], 'targeted_data')
    }

    train_prection_model_config = {
        'train_source_path': f'{os.path.join(targeting_config.get('output_path', f'sample_data/{basic_config['reports_name']}/targeted_data'), 'train_labeled.parquet')}',
        'val_source_path': f'{os.path.join(targeting_config.get('output_path', f'sample_data/{basic_config['reports_name']}/targeted_data'), 'val_labeled.parquet')}',
        'test_source_path': f'{os.path.join(targeting_config.get('output_path', f'sample_data/{basic_config['reports_name']}/targeted_data'), 'test_labeled.parquet')}',
    }

    welcome_message(basic_config)

    data_wrangling(data_wrangling_config)

    feature_enginiering(feature_enginiering_config)

    data_split(data_split_config)

    targerting(targeting_config)

if __name__ == "__main__":
    main()