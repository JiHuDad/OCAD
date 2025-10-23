# 데이터 디렉토리 구조 정리

**날짜**: 2025-10-23
**이유**: `ocad/data/`와 `data/` 두 곳에 데이터가 흩어져 혼란 발생

## 문제점

기존 구조에서 데이터가 두 곳에 분산되어 있었습니다:

```
OCAD/
├── data/                       # 프로젝트 루트 데이터
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── samples/                # 샘플 데이터
│
└── ocad/                       # Python 패키지
    └── data/                   # ❌ 혼란!
        └── training/           # 학습 데이터
```

**문제:**
- ❌ 데이터 위치가 불명확 (프로젝트 루트? 패키지 내부?)
- ❌ 코드와 데이터가 섞여 있음 (`ocad/`는 코드 패키지인데 데이터 포함)
- ❌ `.gitignore` 관리 어려움
- ❌ 배포 시 혼란 (패키지에 데이터 포함?)

## 해결 방법

**원칙**: 모든 데이터는 프로젝트 루트의 `data/`에, `ocad/`는 순수 코드만

### 정리 작업

```bash
# 1. ocad/data/training -> data/training 이동
mkdir -p /home/finux/dev/OCAD/data/training
mv /home/finux/dev/OCAD/ocad/data/training/* /home/finux/dev/OCAD/data/training/

# 2. 빈 디렉토리 제거
rmdir /home/finux/dev/OCAD/ocad/data/training
rmdir /home/finux/dev/OCAD/ocad/data
```

## 정리 후 구조

```
OCAD/
├── data/                       # ✅ 모든 데이터 (프로젝트 루트)
│   ├── training/               # 학습 데이터
│   │   ├── timeseries_train.parquet
│   │   ├── timeseries_val.parquet
│   │   └── timeseries_test.parquet
│   ├── samples/                # 샘플 데이터
│   │   ├── 01_normal_operation_24h.csv
│   │   ├── 02_drift_anomaly.csv/.xlsx
│   │   ├── 03_spike_anomaly.csv
│   │   ├── 04_multi_endpoint.csv/.parquet
│   │   ├── 05_weekly_data.parquet
│   │   └── 06_comprehensive_example.xlsx
│   ├── raw/                    # 원본 데이터
│   ├── processed/              # 처리된 데이터
│   └── synthetic/              # 합성 데이터
│
└── ocad/                       # ✅ Python 코드만
    ├── loaders/                # 파일 로더
    ├── api/                    # REST API
    ├── core/                   # 핵심 모듈
    ├── training/               # 학습 로직 (코드만!)
    └── ...
```

## 장점

✅ **명확한 분리**: 데이터는 `data/`, 코드는 `ocad/`
✅ **찾기 쉬움**: 모든 데이터는 한 곳에
✅ **배포 용이**: `ocad/` 패키지에 데이터 없음
✅ **`.gitignore` 관리**: `data/` 전체를 제외하기 쉬움
✅ **일관성**: Python 프로젝트 모범 사례 준수

## 영향받는 스크립트

정리 후 경로가 변경되었지만, 대부분의 스크립트는 이미 프로젝트 루트의 `data/`를 사용하고 있어 영향 없음:

### 영향 없음 ✅

```python
# 이미 올바른 경로 사용
scripts/generate_sample_data.py          # data/samples/
scripts/generate_training_data.py        # data/training/ (이미 수정됨)
scripts/test_file_loaders.py             # data/samples/
```

### 확인 필요

만약 `ocad/data/training/`을 직접 참조하는 코드가 있다면:

```python
# Before
from pathlib import Path
data_path = Path(__file__).parent / "ocad" / "data" / "training"

# After
data_path = Path(__file__).parent / "data" / "training"
```

또는 프로젝트 루트 기준:

```python
# 권장 방법
from pathlib import Path
project_root = Path(__file__).parent.parent  # 또는 적절한 레벨
data_path = project_root / "data" / "training"
```

## 검증

정리 후 확인:

```bash
# 1. ocad/data/ 디렉토리가 없는지 확인
ls /home/finux/dev/OCAD/ocad/data/
# 예상: "그런 파일이나 디렉터리가 없습니다"

# 2. data/training/ 파일 확인
ls -lh /home/finux/dev/OCAD/data/training/
# 예상: timeseries_train.parquet, timeseries_val.parquet, timeseries_test.parquet

# 3. 학습 스크립트 테스트
python scripts/train_tcn_model.py --help
# 예상: 정상 동작

# 4. 파일 로더 테스트
python scripts/test_file_loaders.py
# 예상: 6개 테스트 모두 통과
```

## 참고 문서

- [CLAUDE.md](../CLAUDE.md) - 업데이트된 디렉토리 구조 포함
- [Quick-Start-Guide.md](Quick-Start-Guide.md) - 데이터 위치 안내
- [Model-Training-Guide.md](Model-Training-Guide.md) - 학습 데이터 경로

---

**작성자**: Claude Code
**최종 업데이트**: 2025-10-23
**상태**: ✅ 완료
