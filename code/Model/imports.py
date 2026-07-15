"""机器学习工程统一依赖入口；业务模块只从这里导入。"""
from __future__ import annotations

import argparse
import contextlib
import gc
import html
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import sys
import subprocess
import threading
import queue
import re
import time
import traceback
import warnings
from typing import Any, Iterable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:  # pragma: no cover
    optuna = None
    TPESampler = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None
