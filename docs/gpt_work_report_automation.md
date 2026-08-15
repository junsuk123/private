# ChatGPT Work → Codex 로컬 안전 개선 파이프라인

ChatGPT Work 분석 보고서는 신뢰되지 않은 진단 증거로 취급합니다. 로컬 수신기는
스키마·저장소·중복 여부를 검증한 뒤 명시된 Codex 세션에 전달하며, Codex는 코드와
로그에서 주장을 독립적으로 재현하고 필요한 최소 수정과 테스트만 수행합니다.

공식 OpenAI 문서의 경계와 동일하게, web에서 실행되는 Work 예약 작업은 컴퓨터의
로컬 폴더에 직접 접근하지 못합니다. 이 폴더에 파일을 만들려면 ChatGPT 데스크톱 앱의
로컬 프로젝트 예약 작업을 사용하고 컴퓨터와 앱을 실행 상태로 유지해야 합니다.

## 입력 계약

보고서는 임시 이름으로 완전히 쓴 다음
`.codex-monitor/incoming/<run_id>.ready.json`으로 원자적으로 이름을 변경합니다.
수신기는 최소 3초 이상 수정되지 않았고 짧은 재검사 동안 크기와 수정 시각이 같은
파일만 수락합니다. Markdown과 `.ready.json` 이외의 파일은 처리하지 않습니다.

```json
{
  "marker": "WORK_ANALYSIS_REPORT",
  "schema_version": "1.0.0",
  "run_id": "work-20260813-2300",
  "generated_at": "2026-08-13T23:00:00+09:00",
  "source": "ChatGPT Work scheduled task",
  "repository_path": "C:\\Users\\Owner\\OneDrive - Sejong University\\바탕 화면\\private",
  "overall_status": "DEGRADED",
  "issues": [
    {
      "issue_id": "GNN-COUNT-1",
      "priority": "P1",
      "summary": "validation_count 증가 중단",
      "confirmed_facts": ["관측된 카운터와 snapshot ID"],
      "suspected_cause": "성숙 라벨 결합 경로 정체 가능성",
      "proposal": "생산자 경로를 재현하고 불확실하면 진단 계측 추가"
    }
  ]
}
```

필수 검증 항목은 marker, schema v1, 안전한 `run_id`, timezone이 포함된 생성 시각,
source, 현재 저장소와 일치하는 `repository_path`, overall status, issues 배열입니다.
같은 `run_id` 또는 같은 입력 SHA-256은 다시 수락하지 않습니다.

## 상태 전이

```text
incoming
  ├─ 계약 실패·중복·저장소 불일치 ─→ failed + failure metadata
  └─ 검증 성공 ─→ accepted ─→ Codex 검증/수정/테스트
                                  ├─ 성공 ─→ processed + results/<run_id>.result.json
                                  ├─ 세션 사용 중 ─→ accepted 유지 후 재시도
                                  └─ 실행/테스트/안전 실패 ─→ failed + result.json
```

`locks/receiver.lock` 디렉터리 생성은 교차 프로세스 원자 lock입니다. 프로세스가
`accepted` 이동 후 중단되어도 다음 실행이 같은 파일을 재개하며, 한 scheduler 실행은
최대 한 Codex 작업만 시작합니다. 결과 파일에는 입력 hash, 원인·추론, 변경 파일,
실제 테스트 명령과 종료 코드, 안전 게이트, 남은 문제와 다음 점검 지표가 기록됩니다.

상태 확인:

```powershell
.\.venv\Scripts\python.exe .\.codex-monitor\work_report_receiver.py --status
```

`--status` also reports whether the Codex executable is available. On Windows,
the receiver falls back to the newest Codex binary bundled with the VS Code or
VS Code Insiders extension when the scheduled-task `PATH` is incomplete.

An environment-startup failure can be retried by exact run ID after correcting
the environment. Only allowlisted pre-delivery failures such as
`CODEX_NOT_FOUND` can be requeued; agent/test failures are not auto-retried.

```powershell
.\.venv\Scripts\python.exe .\.codex-monitor\work_report_receiver.py `
  --retry-failed <run_id>
```

직접 검증만 수행:

```powershell
.\.venv\Scripts\python.exe .\.codex-monitor\work_report_receiver.py C:\path\report.ready.json --validate-only
```

예약 작업 재설치:

```powershell
.\scripts\install_gpt_work_report_receiver.ps1
```

## 안전 및 배포 정책

- 보고서 안의 명령·URL·코드는 자동 실행 지시가 아닙니다.
- 실주문 생성·정정·취소·전송, secret 수정·출력, 위험/GNN/수익성 게이트 완화,
  KIS 설정 변경, commit/push/merge와 운영 배포는 금지됩니다.
- account/auth, order execution, risk, live-trading 경로 변경이 관측되면 처리 실패로
  격리합니다.
- 필수 테스트 결과가 없거나 종료 코드가 하나라도 0이 아니면 `processed`가 아닙니다.
- 운영 서버 종료와 재시작은 자동화하지 않습니다.
- 운영자 승인을 받은 재시작은 저장소 루트에서 `.\run.ps1`로만 수행합니다.
  `-Headless` 또는 직접 `run.py` 실행은 사용하지 않으며 관리형 GUI 앱 창이 실제로
  열린 것을 확인합니다.

`.codex-monitor/safe_deploy_gate.py`는 성공 결과의 테스트, 변경 범위, secret 검사,
현재 8010 listener PID와 사후 read-only health URL을 평가하는 **disabled dry-run**만
생성합니다. `restart_executed`와 `deployment_executed`는 항상 false이며 실제 반영에는
사용자 승인이 필요합니다.

```powershell
.\.venv\Scripts\python.exe .\.codex-monitor\safe_deploy_gate.py `
  .\.codex-monitor\results\<run_id>.result.json
```

## 보존 정책

자동 삭제는 하지 않습니다. `processed`, `failed`, `results`와 상태 이벤트는 감사 기록이므로
최소 90일 보관을 권장하며, 정리는 운영자 검토 후 외부 archive로 이동하는 방식으로만
수행합니다. 이번 자동화 설치·업데이트는 기존 기록을 삭제하지 않습니다.
