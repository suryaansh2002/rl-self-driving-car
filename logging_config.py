import logging
import os
import sys

def setup_logger(logger_name, log_files=None, console_output=False):
    """
    Sets up a logger with specified name and log files
    
    Args:
        logger_name (str): Name of the logger
        log_files (list): List of log file paths. If None, uses default log file
        console_output (bool): Whether to also output logs to console
    """
    try:
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs', exist_ok=True)
            
        # Get logger
        logger = logging.getLogger(logger_name)
        
        # Only set up handlers if they don't exist
        if not logger.handlers:
            # Set logging level
            logger.setLevel(logging.INFO)
            
            # Default log file if none specified
            if log_files is None:
                log_files = [f'logs/{logger_name.lower()}.log']
                
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Add console handler if requested
            if console_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # Add file handlers
            for log_file in log_files:
                try:
                    file_handler = logging.FileHandler(log_file, mode='a')
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    print(f"Error creating log file {log_file}: {str(e)}")
                    
        return logger
        
    except Exception as e:
        print(f"Error setting up logger {logger_name}: {str(e)}")
        return None