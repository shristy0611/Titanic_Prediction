import json_log_formatter
import logging

formatter = json_log_formatter.JSONFormatter()
json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)

logger = logging.getLogger('titanic_service')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)
