#!/usr/bin/env python3
"""
Rate-based CAN IDS detector.

Detects anomalies based on:
1. Message rate deviations per CAN ID
2. Unusual timing patterns (too regular or too irregular)
3. Unexpected ID appearances
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import time


@dataclass
class RateDetectorConfig:
    """Configuration for rate-based detection"""
    # Track rates over this many seconds
    history_window_seconds: float = 30.0
    
    # Short-term window for faster adaptation (used for recent baseline)
    short_term_window_seconds: float = 10.0
    
    # Minimum samples needed before alerting
    min_samples: int = 5
    
    # Alert if rate deviates by this many standard deviations
    rate_deviation_threshold: float = 3.0
    
    # Alert if rate is above this multiplier of mean
    rate_multiplier_threshold: float = 2.0
    
    # Alert if rate is below this fraction of mean (for low-volume attacks)
    rate_minimum_threshold: float = 0.1
    
    # Alert if timing regularity is too high (coefficient of variation < threshold)
    regularity_threshold: float = 0.1  # CV < 0.1 means very regular
    
    # Alert if timing is too irregular (coefficient of variation > threshold)
    irregularity_threshold: float = 2.0
    
    # Track these specific IDs (if empty, track all)
    monitored_ids: List[int] = field(default_factory=list)
    
    # Ignore these IDs
    ignored_ids: List[int] = field(default_factory=list)


@dataclass
class IDRateStats:
    """Statistics for a single CAN ID"""
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10000))
    rates: deque = field(default_factory=lambda: deque(maxlen=1000))  # rates per window
    
    def add_message(self, timestamp: float) -> None:
        """Add a message timestamp"""
        self.timestamps.append(timestamp)
    
    def update_rate(self, rate: float) -> None:
        """Update rate for current window"""
        self.rates.append(rate)
    
    def get_recent_timestamps(self, window_seconds: float, current_time: float) -> List[float]:
        """Get timestamps within the window"""
        cutoff = current_time - window_seconds
        return [ts for ts in self.timestamps if ts >= cutoff]
    
    def compute_rate_stats(self, window_seconds: float, current_time: float, short_term_window: float = None) -> Tuple[float, float, float]:
        """Compute mean, std, and current rate"""
        recent = self.get_recent_timestamps(window_seconds, current_time)
        if len(recent) < 2:
            return 0.0, 0.0, 0.0
        
        # Compute rate as messages per second
        if len(recent) >= 2:
            duration = recent[-1] - recent[0]
            if duration > 0:
                current_rate = (len(recent) - 1) / duration
            else:
                current_rate = len(recent)
        else:
            current_rate = 0.0
        
        # Use weighted average: recent rates weighted more heavily for faster adaptation
        if len(self.rates) >= 2:
            # Use exponential weighting: more recent rates have higher weight
            weights = np.exp(np.linspace(-2, 0, len(self.rates)))  # Recent = higher weight
            weights = weights / weights.sum()
            
            mean_rate = float(np.average(self.rates, weights=weights))
            
            # For std, use recent rates (last 50% of history) for faster adaptation
            recent_rates = self.rates[-max(5, len(self.rates) // 2):]
            std_rate = float(np.std(recent_rates)) if len(recent_rates) > 1 else mean_rate * 0.1
        else:
            # Fallback to current window estimate
            mean_rate = current_rate
            std_rate = current_rate * 0.1  # Assume 10% variation
        
        return mean_rate, std_rate, current_rate
    
    def compute_timing_regularity(self, window_seconds: float, current_time: float) -> float:
        """Compute coefficient of variation of inter-arrival times (lower = more regular)"""
        recent = self.get_recent_timestamps(window_seconds, current_time)
        if len(recent) < 3:
            return 1.0  # Not enough data
        
        intervals = np.diff(sorted(recent))
        if len(intervals) == 0 or np.mean(intervals) == 0:
            return 1.0
        
        cv = np.std(intervals) / np.mean(intervals)  # Coefficient of variation
        return float(cv)


class RateDetector:
    """Rate-based anomaly detector"""
    
    def __init__(self, config: RateDetectorConfig):
        self.config = config
        self.id_stats: Dict[int, IDRateStats] = defaultdict(IDRateStats)
        self.window_start_time: Optional[float] = None
        self.window_message_counts: Dict[int, int] = defaultdict(int)
    
    def add_frame(self, can_id: int, timestamp: float) -> None:
        """Add a CAN frame to the detector"""
        # Skip ignored IDs
        if can_id in self.config.ignored_ids:
            return
        
        # Track all IDs or only monitored ones
        if self.config.monitored_ids and can_id not in self.config.monitored_ids:
            return
        
        # Add to statistics
        self.id_stats[can_id].add_message(timestamp)
        self.window_message_counts[can_id] += 1
    
    def check_anomalies(self, current_time: float, window_duration: float) -> Dict[str, any]:
        """
        Check for rate-based anomalies.
        
        Returns dict with:
        - is_anomaly: bool
        - score: float (0-1, higher = more anomalous)
        - alerts: List of alert descriptions
        """
        alerts = []
        max_score = 0.0
        
        # Check each monitored ID
        for can_id, stats in self.id_stats.items():
            # Use shorter window for faster adaptation
            short_term = self.config.short_term_window_seconds if hasattr(self.config, 'short_term_window_seconds') else self.config.history_window_seconds / 3
            
            mean_rate, std_rate, current_rate = stats.compute_rate_stats(
                self.config.history_window_seconds, current_time, short_term
            )
            
            # Skip if not enough data (need at least some history)
            if len(stats.timestamps) < max(3, self.config.min_samples // 2):
                continue
            
            # Check for rate deviations
            if mean_rate > 0:
                # High rate deviation
                if std_rate > 0:
                    z_score = abs(current_rate - mean_rate) / std_rate
                    if z_score > self.config.rate_deviation_threshold:
                        alerts.append(
                            f"ID 0x{can_id:03X}: rate anomaly "
                            f"(current={current_rate:.1f} msg/s, mean={mean_rate:.1f}±{std_rate:.1f}, z={z_score:.2f})"
                        )
                        max_score = max(max_score, min(1.0, z_score / self.config.rate_deviation_threshold))
                
                # Rate multiplier check
                if current_rate > mean_rate * self.config.rate_multiplier_threshold:
                    alerts.append(
                        f"ID 0x{can_id:03X}: high rate "
                        f"({current_rate:.1f} msg/s > {mean_rate * self.config.rate_multiplier_threshold:.1f} threshold)"
                    )
                    max_score = max(max_score, 0.8)
                
                # Low rate check (for low-volume injection attacks)
                if current_rate < mean_rate * self.config.rate_minimum_threshold and mean_rate > 10:
                    # Only alert if normal rate is high (not just sparse traffic)
                    alerts.append(
                        f"ID 0x{can_id:03X}: suspiciously low rate "
                        f"({current_rate:.1f} msg/s < {mean_rate * self.config.rate_minimum_threshold:.1f} threshold, "
                        f"normal={mean_rate:.1f} msg/s)"
                    )
                    max_score = max(max_score, 0.7)
            
            # Check timing regularity (detect injection patterns)
            # Use shorter window for faster adaptation
            short_term = self.config.short_term_window_seconds if hasattr(self.config, 'short_term_window_seconds') else self.config.history_window_seconds / 3
            cv = stats.compute_timing_regularity(short_term, current_time)
            
            # Build baseline CV using recent history (faster adaptation)
            if len(stats.timestamps) >= 10:  # Reduced from 20 for faster detection
                # Use shorter windows for baseline (more recent data)
                recent_cvs = []
                num_windows = min(5, len(stats.timestamps) // 3)  # Fewer windows, faster
                for i in range(num_windows):
                    window_start = current_time - (i + 1) * short_term / num_windows
                    window_cv = stats.compute_timing_regularity(
                        short_term / num_windows, window_start
                    )
                    if window_cv < 1.0:  # Valid CV
                        recent_cvs.append(window_cv)
                
                if len(recent_cvs) >= 3:  # Reduced from 5
                    # Weight recent CVs more heavily
                    weights = np.exp(np.linspace(-1, 0, len(recent_cvs)))
                    weights = weights / weights.sum()
                    baseline_cv = float(np.average(recent_cvs, weights=weights))
                    cv_std = float(np.std(recent_cvs[-3:])) if len(recent_cvs) >= 3 else baseline_cv * 0.1
                    
                    # Only alert if current CV is significantly lower than baseline
                    # AND below absolute threshold (to catch truly suspicious patterns)
                    if cv < self.config.regularity_threshold and cv < baseline_cv - 1.5 * cv_std:  # Reduced from 2*std for faster detection
                        alerts.append(
                            f"ID 0x{can_id:03X}: suspiciously regular timing "
                            f"(CV={cv:.3f} < baseline={baseline_cv:.3f}±{cv_std:.3f}, threshold={self.config.regularity_threshold})"
                        )
                        max_score = max(max_score, 0.9)
            elif cv < self.config.regularity_threshold and len(stats.timestamps) >= 5:  # Reduced from 10
                # Fallback: alert if CV is extremely low (near zero) even without baseline
                if cv < 0.005:  # Very strict - only near-perfect regularity
                    alerts.append(
                        f"ID 0x{can_id:03X}: extremely regular timing "
                        f"(CV={cv:.3f} < {self.config.regularity_threshold} threshold)"
                    )
                    max_score = max(max_score, 0.7)
            
            # Use shorter window for irregularity check too
            cv_long = stats.compute_timing_regularity(self.config.history_window_seconds, current_time)
            if cv_long > self.config.irregularity_threshold and len(stats.timestamps) >= 5:
                alerts.append(
                    f"ID 0x{can_id:03X}: highly irregular timing "
                    f"(CV={cv_long:.3f} > {self.config.irregularity_threshold} threshold)"
                )
                max_score = max(max_score, 0.6)
        
        # Update window rates (limit history size for faster adaptation)
        if window_duration > 0:
            for can_id, count in self.window_message_counts.items():
                window_rate = count / window_duration
                stats = self.id_stats[can_id]
                stats.update_rate(window_rate)
                
                # Limit rate history to recent windows only (faster adaptation)
                max_rate_history = int(self.config.history_window_seconds / window_duration) + 10
                if len(stats.rates) > max_rate_history:
                    # Keep only most recent rates
                    stats.rates = deque(list(stats.rates)[-max_rate_history:], maxlen=stats.rates.maxlen)
        
        # Reset window counters
        self.window_message_counts.clear()
        
        return {
            "is_anomaly": len(alerts) > 0,
            "score": max_score,
            "alerts": alerts,
            "details": {
                "checked_ids": len(self.id_stats),
                "alert_count": len(alerts)
            }
        }
    
    def reset_window(self, start_time: float) -> None:
        """Reset window counters"""
        self.window_start_time = start_time
        self.window_message_counts.clear()

