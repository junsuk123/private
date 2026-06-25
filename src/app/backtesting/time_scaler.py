"""
시간 스케일링 추상화 계층 - 데모/테스트 모드에서 시간을 가속화할 수 있도록 함.

이 모듈은 실제 시간 대신 가상 시간을 관리하여 10배 가속 등을 지원합니다.
나중에 실제 API로 전환할 때도 이 인터페이스를 유지하면 되므로 분리가 용이합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


class TimeMode(Enum):
    """시간 실행 모드"""
    REALTIME = "realtime"  # 1배속 (실제 시간)
    ACCELERATED = "accelerated"  # 배속 모드 (10배, 100배 등)


@dataclass(frozen=True)
class TimeScalerConfig:
    """시간 스케일러 설정"""
    mode: TimeMode = TimeMode.ACCELERATED
    acceleration_factor: float = 10.0  # 10배 가속
    start_time: Optional[datetime] = None  # None이면 현재 시간


class TimeScaler:
    """
    시간 스케일링을 관리하는 클래스.
    
    데모 모드에서는 가상 시간이 실제보다 빠르게 진행되고,
    실제 운영에서는 1배속으로 운영됩니다.
    
    예시:
        scaler = TimeScaler(TimeScalerConfig(
            mode=TimeMode.ACCELERATED,
            acceleration_factor=10.0
        ))
        
        # 1초 지나면 가상으로 10초가 지남
        virtual_time = scaler.get_virtual_time()
        time.sleep(1)
        virtual_time_after = scaler.get_virtual_time()
        # virtual_time_after - virtual_time ≈ 10 seconds
    """

    def __init__(self, config: TimeScalerConfig) -> None:
        self.config = config
        self._reference_real_time = datetime.now(timezone.utc)
        self._reference_virtual_time = config.start_time or datetime.now(timezone.utc)
        self._is_paused = False
        self._pause_virtual_time: Optional[datetime] = None

    def get_virtual_time(self) -> datetime:
        """현재 가상 시간을 반환합니다."""
        if self._is_paused:
            return self._pause_virtual_time or self._reference_virtual_time

        if self.config.mode == TimeMode.REALTIME:
            return datetime.now(timezone.utc)
        
        # ACCELERATED 모드
        elapsed_real = datetime.now(timezone.utc) - self._reference_real_time
        elapsed_virtual = timedelta(
            seconds=elapsed_real.total_seconds() * self.config.acceleration_factor
        )
        return self._reference_virtual_time + elapsed_virtual

    def advance_virtual_time(self, delta: timedelta) -> datetime:
        """가상 시간을 지정된 시간만큼 진행시킵니다."""
        current = self.get_virtual_time()
        new_time = current + delta
        
        # 기준점을 업데이트하여 다음 호출에서 정확한 시간을 반환하도록 함
        self._reference_virtual_time = new_time
        self._reference_real_time = datetime.now(timezone.utc)
        
        return new_time

    def pause(self) -> None:
        """가상 시간을 일시 정지합니다."""
        self._is_paused = True
        self._pause_virtual_time = self.get_virtual_time()

    def resume(self) -> None:
        """가상 시간을 다시 진행합니다."""
        if self._is_paused:
            self._is_paused = False
            self._reference_real_time = datetime.now(timezone.utc)
            self._reference_virtual_time = self._pause_virtual_time or self.get_virtual_time()
            self._pause_virtual_time = None

    def reset(self, start_time: Optional[datetime] = None) -> None:
        """시간 스케일러를 리셋합니다."""
        self._reference_real_time = datetime.now(timezone.utc)
        self._reference_virtual_time = start_time or datetime.now(timezone.utc)
        self._is_paused = False
        self._pause_virtual_time = None

    def get_config(self) -> TimeScalerConfig:
        """현재 설정을 반환합니다."""
        return self.config

    def is_paused(self) -> bool:
        """일시 정지 상태인지 확인합니다."""
        return self._is_paused

    def get_scale_factor(self) -> float:
        """현재 가속 배율을 반환합니다 (1.0 = 1배속, 10.0 = 10배속)."""
        if self.config.mode == TimeMode.REALTIME:
            return 1.0
        return self.config.acceleration_factor
