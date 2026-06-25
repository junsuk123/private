@echo off
if "%PYTHONPATH%"=="" set PYTHONPATH=src
if "%APP_ENV%"=="" set APP_ENV=local
if "%DATA_ENV%"=="" set DATA_ENV=live
if "%TRADING_MODE%"=="" set TRADING_MODE=read_only
if "%LIVE_TRADING_ENABLED%"=="" set LIVE_TRADING_ENABLED=false
if "%ONTOLOGY_ACCELERATOR%"=="" set ONTOLOGY_ACCELERATOR=NPU
if "%REALTIME_LATENCY_PROFILE%"=="" set REALTIME_LATENCY_PROFILE=low_latency
if "%OPENVINO_DEVICE%"=="" set OPENVINO_DEVICE=NPU
if "%OPENVINO_HINT_PERFORMANCE_MODE%"=="" set OPENVINO_HINT_PERFORMANCE_MODE=LATENCY
if "%OPENVINO_ENABLE_CPU_PINNING%"=="" set OPENVINO_ENABLE_CPU_PINNING=YES
if "%LLM_EVENT_PROVIDER%"=="" if exist "models\local-llm\event-classifier" set LLM_EVENT_PROVIDER=embedded
if "%LLM_EVENT_PROVIDER%"=="" set LLM_EVENT_PROVIDER=local
if "%LLM_EVENT_MODEL%"=="" if "%LLM_EVENT_PROVIDER%"=="embedded" set LLM_EVENT_MODEL=models\local-llm\event-classifier
if "%LLM_EVENT_MODEL%"=="" set LLM_EVENT_MODEL=qwen2.5:1.5b-instruct
if "%LLM_EVENT_PROVIDER%"=="embedded" if "%LLM_EVENT_MODEL_CACHE_DIR%"=="" set LLM_EVENT_MODEL_CACHE_DIR=models\local-llm\cache
if "%LLM_EVENT_PROVIDER%"=="embedded" if "%LLM_EVENT_LOCAL_FILES_ONLY%"=="" set LLM_EVENT_LOCAL_FILES_ONLY=true
if "%LLM_EVENT_PROVIDER%"=="embedded" if "%LLM_EVENT_DEVICE%"=="" set LLM_EVENT_DEVICE=auto
if "%LLM_EVENT_PROVIDER%"=="multimodal" if "%LLM_EVENT_LOCAL_FILES_ONLY%"=="" set LLM_EVENT_LOCAL_FILES_ONLY=true
if "%LLM_EVENT_PROVIDER%"=="multimodal" if "%LLM_EVENT_DEVICE%"=="" set LLM_EVENT_DEVICE=auto
if "%LLM_EVENT_LOCAL_ENDPOINT%"=="" set LLM_EVENT_LOCAL_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions
if "%LLM_EVENT_CLASSIFIER_ENABLED%"=="" if "%LLM_EVENT_PROVIDER%"=="multimodal" set LLM_EVENT_CLASSIFIER_ENABLED=true
if "%LLM_EVENT_CLASSIFIER_ENABLED%"=="" if not "%LLM_EVENT_PROVIDER%"=="multimodal" set LLM_EVENT_CLASSIFIER_ENABLED=false
if "%LIVE_REFRESH_SECONDS%"=="" set LIVE_REFRESH_SECONDS=15
if "%LEARNING_COLLECTION_INTERVAL_SECONDS%"=="" set LEARNING_COLLECTION_INTERVAL_SECONDS=60
if "%RESEARCH_RETENTION_DAYS%"=="" set RESEARCH_RETENTION_DAYS=30
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py --port 8000 --strict-port
) else (
  python run.py --port 8000 --strict-port
)
