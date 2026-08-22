import { useState, useEffect, useCallback, useRef } from "react"
import { Header } from "@/components/layout/Header"
import { Sidebar } from "@/components/layout/Sidebar"
import { StatusBar } from "@/components/layout/StatusBar"
import { FileUploader } from "@/components/data/FileUploader"
import { TeacherModelSelector } from "@/components/data/TeacherModelSelector"
import { ModelGrid } from "@/components/model/ModelGrid"
import { TrainingControl } from "@/components/training/TrainingControl"
import { ProgressTracker } from "@/components/training/ProgressTracker"
import { LogViewer } from "@/components/training/LogViewer"
import { SystemMetrics } from "@/components/monitor/SystemMetrics"
import { ExportPanel } from "@/components/export/ExportPanel"
import { TrainingParams } from "@/components/pro/TrainingParams"
import { DataManager } from "@/components/pro/DataManager"
import { TrainingMonitor } from "@/components/pro/TrainingMonitor"
import { ModelManager } from "@/components/pro/ModelManager"
import { OneClickPipeline } from "@/components/pro/OneClickPipeline"
import { useHardware } from "@/hooks/useHardware"
import { modelPresets, resolveHfModel, sizeTierFor } from "@/components/model/ModelPresets"
import type { UploadedFile, Metrics, ViewMode, ActiveView, Job, JobStage, TrainingParams as TrainingParamsType, ModelVersion } from "@/types"

// In dev the Vite server runs on :5173 and proxies nothing, so hit the backend
// directly; in production the built bundle is served by FastAPI itself.
const API_BASE = import.meta.env.DEV ? "http://127.0.0.1:8080" : ""

const defaultTrainingParams: TrainingParamsType = {
  loraR: 8,
  loraAlpha: 16,
  loraDropout: 0.05,
  learningRate: 0.0002,
  epochs: 3,
  batchSize: 4,
  gradientAccumulation: 4,
  maxSeqLength: 512,
  scheduler: 'cosine'
}

const sampleTrainingData = [
  { id: '1', prompt: '什么是模型蒸馏？', response: '模型蒸馏是一种将大模型（教师）的知识转移到小模型（学生）的技术。' },
  { id: '2', prompt: 'LoRA是什么？', response: 'LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，通过低秩分解减少可训练参数。' },
]

// Backend pipeline stages in display order (vision-only stages omitted).
const STAGE_FLOW: Array<{ id: JobStage; name: string }> = [
  { id: 'prepare', name: '环境准备' },
  { id: 'data', name: '数据处理' },
  { id: 'synthetic', name: '数据合成' },
  { id: 'train', name: '模型训练' },
  { id: 'quantize', name: '量化导出' },
]

function App() {
  const { hardware } = useHardware()
  const [connected, setConnected] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('simple')
  const [activeView, setActiveView] = useState<ActiveView>('data')
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [teacherProvider, setTeacherProvider] = useState("deepseek")
  const [teacherModel, setTeacherModel] = useState("deepseek-chat")
  const [apiKey, setApiKey] = useState("")
  const [exportFormat, setExportFormat] = useState("gguf")
  const [quantization, setQuantization] = useState("q4_k_m")
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [metricsHistory, setMetricsHistory] = useState<Metrics[]>([])
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [isExporting, setIsExporting] = useState(false)

  const [trainingParams, setTrainingParams] = useState<TrainingParamsType>(defaultTrainingParams)
  const [trainingData, setTrainingData] = useState(sampleTrainingData)
  const [trainingHistory] = useState<Array<{
    epoch: number
    step: number
    loss: number
    learningRate: number
    tokensPerSecond?: number
  }>>([])
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([])
  const jobWsRef = useRef<WebSocket | null>(null)

  useEffect(() => () => jobWsRef.current?.close(), [])

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`)
        setConnected(response.ok)
      } catch {
        setConnected(false)
      }
    }

    checkConnection()
    const interval = setInterval(checkConnection, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!connected) return

    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/metrics`)
        if (response.ok) {
          const data = await response.json()
          const newMetrics: Metrics = {
            cpu: typeof data.cpu_percent === 'number' ? data.cpu_percent : 0,
            ram: data.ram_total_gb
              ? Math.min(100, Math.round((data.ram_used_gb / data.ram_total_gb) * 1000) / 10)
              : 0,
            gpu: undefined,
            vram: typeof data.vram_used_gb === 'number' ? data.vram_used_gb : undefined,
            timestamp: new Date()
          }
          setMetrics(newMetrics)
          setMetricsHistory(prev => [...prev.slice(-59), newMetrics])
        }
      } catch (error) {
        console.error("Failed to fetch metrics:", error)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 2000)
    return () => clearInterval(interval)
  }, [connected])

  const appendLog = useCallback((line: string) => {
    setLogs(prev => {
      const next = prev.length >= 300 ? prev.slice(-250) : [...prev]
      if (next[next.length - 1] === line) return prev
      next.push(line)
      return next
    })
  }, [])

  // Load completed jobs from previous sessions into the model manager.
  useEffect(() => {
    if (!connected) return
    ;(async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/jobs`)
        if (!resp.ok) return
        const jobs = (await resp.json()) as Job[]
        setModelVersions(jobs
          .filter(j => j.status === 'done')
          .map(j => ({
            id: j.id,
            name: String((j.spec as Record<string, unknown> | undefined)?.model ?? `job-${j.id}`),
            version: new Date(j.created_at * 1000).toISOString().slice(0, 10),
            size: Number(j.metrics?.size_mb ?? 0) * 1024 * 1024,
            format: 'GGUF',
            quantization: '-',
            createdAt: new Date(j.created_at * 1000),
          })))
      } catch { /* backend history is optional */ }
    })()
  }, [connected])

  const handleFilesAdd = useCallback(async (newFiles: File[]) => {
    const items: UploadedFile[] = newFiles.map(file => ({
      id: Math.random().toString(36).slice(2, 11),
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: 'uploading'
    }))
    setFiles(prev => [...prev, ...items])
    await Promise.all(items.map(async (item, i) => {
      try {
        const fd = new FormData()
        fd.append('file', newFiles[i])
        const resp = await fetch(`${API_BASE}/api/files`, { method: 'POST', body: fd })
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        setFiles(prev => prev.map(f =>
          f.id === item.id ? { ...f, path: data.path, progress: 100, status: 'ready' } : f))
      } catch {
        setFiles(prev => prev.map(f =>
          f.id === item.id ? { ...f, status: 'error' } : f))
      }
    }))
  }, [])

  const handleFileRemove = useCallback((id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id))
  }, [])

  // Subscribe to the backend's live job stream over WebSocket.
  const trackJob = useCallback((jobId: string) => {
    jobWsRef.current?.close()
    const wsBase = API_BASE.replace(/^http/, 'ws')
    const ws = new WebSocket(`${wsBase}/ws/jobs/${jobId}`)
    ws.onmessage = ev => {
      try {
        const st = JSON.parse(ev.data) as Job
        setJob(st)
        appendLog(`[${st.stage}] ${st.message}`)
        if (st.status === 'done') {
          const gguf = st.result?.gguf ? `，GGUF：${st.result.gguf}` : ''
          appendLog(`[SUCCESS] 蒸馏完成 ✓${gguf}`)
        } else if (st.status === 'failed') {
          appendLog(`[ERROR] 任务失败: ${st.error ?? st.message}`)
        }
      } catch { /* malformed frame */ }
    }
    ws.onclose = () => {
      // Fetch the final state in case the socket closed before terminal update.
      fetch(`${API_BASE}/api/jobs/${jobId}`)
        .then(r => (r.ok ? r.json() : null))
        .then(st => { if (st && st.id) setJob(st) })
        .catch(() => undefined)
    }
    jobWsRef.current = ws
  }, [appendLog])

  const startTraining = useCallback(async (override?: { model?: string | null }) => {
    const paths = files.map(f => f.path).filter((p): p is string => !!p)
    const modelId = override?.model !== undefined ? override.model : selectedModel
    const body = {
      source: 'ui',
      task: 'llm',
      data_paths: paths,
      teacher: {
        name: teacherProvider || 'none',
        model: teacherModel || '',
        api_key: apiKey || '',
      },
      model: resolveHfModel(modelId),
      size: sizeTierFor(modelId),
      quantize: true,
    }
    try {
      const response = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      const data = await response.json()
      if (!response.ok) {
        setLogs(prev => [...prev, `[ERROR] 启动失败: ${data.error ?? response.status}`])
        return
      }
      setJob(data)
      setLogs(prev => [...prev,
        `[INFO] 训练任务已启动: ${data.id}（模型=${body.model || '按规格自动'} 规格=${body.size} 数据=${paths.length} 个文件）`])
      trackJob(data.id)
    } catch (error) {
      setLogs(prev => [...prev, `[ERROR] 无法连接后端: ${error}`])
    }
  }, [files, selectedModel, teacherProvider, teacherModel, apiKey, trackJob])

  const handleStartTraining = useCallback(() => {
    void startTraining()
  }, [startTraining])

  const handleCancelTraining = useCallback(async () => {
    if (!job) return
    try {
      const resp = await fetch(`${API_BASE}/api/jobs/${job.id}`, { method: 'DELETE' })
      const data = await resp.json()
      if (!resp.ok) {
        setLogs(prev => [...prev, `[ERROR] 取消失败: ${data.error ?? resp.status}`])
        return
      }
      setJob(data)
      setLogs(prev => [...prev, `[INFO] 已请求取消任务 ${job.id}`])
    } catch (error) {
      setLogs(prev => [...prev, `[ERROR] 取消失败: ${error}`])
    }
  }, [job])

  const hasGgufModel = !!(job && job.status === 'done' && (job.result?.gguf || job.result?.model_dir))

  const handleExport = useCallback(async () => {
    if (!job) return
    setIsExporting(true)
    try {
      const response = await fetch(`${API_BASE}/api/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: job.id, format: exportFormat, quantization })
      })
      const data = await response.json()
      if (!response.ok) {
        setLogs(prev => [...prev, `[ERROR] 导出失败: ${data.error ?? response.status}`])
        return
      }
      setLogs(prev => [...prev, `[SUCCESS] 导出完成: ${data.gguf}（${data.size_mb} MB）`])
    } catch (error) {
      setLogs(prev => [...prev, `[ERROR] 导出失败: ${error}`])
    } finally {
      setIsExporting(false)
    }
  }, [job, exportFormat, quantization])

  const handleOllamaImport = useCallback(async () => {
    const gguf = job?.result?.gguf
    if (!gguf) {
      setLogs(prev => [...prev, '[ERROR] 没有可导入的 GGUF 文件'])
      return
    }
    try {
      const resp = await fetch(`${API_BASE}/api/ollama`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gguf })
      })
      const data = await resp.json()
      if (!resp.ok) {
        setLogs(prev => [...prev, `[ERROR] Ollama 导入失败: ${data.error ?? resp.status}`])
        return
      }
      setLogs(prev => [...prev, `[INFO] 在终端执行以下命令完成导入: ${data.command}`])
    } catch (error) {
      setLogs(prev => [...prev, `[ERROR] Ollama 导入失败: ${error}`])
    }
  }, [job])

  const handleStartServer = useCallback(async () => {
    const gguf = job?.result?.gguf
    if (!gguf) {
      setLogs(prev => [...prev, '[ERROR] 没有可部署的 GGUF 文件'])
      return
    }
    try {
      const resp = await fetch(`${API_BASE}/api/server/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gguf })
      })
      const data = await resp.json()
      if (!resp.ok) {
        setLogs(prev => [...prev, `[ERROR] 启动本地 API 失败: ${data.error ?? resp.status}`])
        return
      }
      setLogs(prev => [...prev, `[SUCCESS] 本地 API 已就绪: ${JSON.stringify(data)}`])
    } catch (error) {
      setLogs(prev => [...prev, `[ERROR] 启动本地 API 失败: ${error}`])
    }
  }, [job])

  const handleDataAdd = useCallback((item: { prompt: string; response: string }) => {
    setTrainingData(prev => [...prev, { ...item, id: Math.random().toString(36).substr(2, 9) }])
  }, [])

  const handleDataUpdate = useCallback((id: string, item: Partial<{ prompt: string; response: string }>) => {
    setTrainingData(prev => prev.map(d => d.id === id ? { ...d, ...item } : d))
  }, [])

  const handleDataRemove = useCallback((id: string) => {
    setTrainingData(prev => prev.filter(d => d.id !== id))
  }, [])

  const handleDeleteModel = useCallback((id: string) => {
    setModelVersions(prev => prev.filter(m => m.id !== id))
  }, [])

  const handleCompareModels = useCallback((ids: string[]) => {
    setLogs(prev => [...prev, `[INFO] 对比模型: ${ids.join(', ')}`])
  }, [])

  const handleOneClickPipeline = useCallback((config: {
    model: string
    params: TrainingParamsType
    autoMode: boolean
  }) => {
    setLogs(prev => [...prev, `[INFO] 启动一键蒸馏: 模型=${config.model}, 自动模式=${config.autoMode}`])
    setTrainingParams(config.params)
    setSelectedModel(config.model)
    // Pass the model explicitly — setState above won't be visible to the
    // in-flight closure this tick.
    void startTraining({ model: config.model })
  }, [startTraining])

  const currentStageIdx = job ? STAGE_FLOW.findIndex(s => s.id === job.stage) : -1
  const doneAll = job?.stage === 'done'
  const trainingStages: Array<{
    id: string
    name: string
    status: 'pending' | 'running' | 'completed' | 'failed'
    progress?: number
  }> = STAGE_FLOW.map((s, i) => {
    let status: 'pending' | 'running' | 'completed' | 'failed' = 'pending'
    if (job) {
      if (i < currentStageIdx || doneAll) status = 'completed'
      else if (i === currentStageIdx) {
        status = job.status === 'done' ? 'completed'
          : job.status === 'failed' && job.stage !== 'prepare' ? 'failed'
          : 'running'
      }
    }
    return {
      id: s.id,
      name: s.name,
      status,
      progress: i === currentStageIdx && !doneAll ? job?.progress : undefined,
    }
  })

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header
        connected={connected}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />
      <div className="flex-1 flex">
        <Sidebar activeView={activeView} onViewChange={setActiveView as (view: string) => void} />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-6xl mx-auto space-y-6">
            {activeView === 'data' && (
              <>
                <FileUploader
                  files={files}
                  onFilesAdd={handleFilesAdd}
                  onFileRemove={handleFileRemove}
                />
                {viewMode === 'pro' && (
                  <>
                    <TeacherModelSelector
                      provider={teacherProvider}
                      model={teacherModel}
                      apiKey={apiKey}
                      onProviderChange={setTeacherProvider}
                      onModelChange={setTeacherModel}
                      onApiKeyChange={setApiKey}
                    />
                    <DataManager
                      data={trainingData}
                      onDataAdd={handleDataAdd}
                      onDataUpdate={handleDataUpdate}
                      onDataRemove={handleDataRemove}
                    />
                  </>
                )}
              </>
            )}

            {activeView === 'model' && (
              <>
                <ModelGrid
                  models={modelPresets}
                  selectedModel={selectedModel}
                  availableRam={hardware?.ram || 16}
                  availableVram={hardware?.vram || null}
                  onSelectModel={setSelectedModel}
                />
                {viewMode === 'pro' && (
                  <ModelManager
                    models={modelVersions}
                    selectedModel={selectedModel}
                    onSelectModel={setSelectedModel}
                    onDeleteModel={handleDeleteModel}
                    onCompareModels={handleCompareModels}
                  />
                )}
              </>
            )}

            {activeView === 'training' && (
              <>
                <OneClickPipeline
                  hardware={hardware}
                  onStartPipeline={handleOneClickPipeline}
                  isRunning={job?.status === 'running' || job?.status === 'queued'}
                />
                <TrainingControl
                  status={
                    job?.status === 'running' || job?.status === 'queued' ? 'running'
                    : job?.status === 'done' ? 'completed'
                    : job?.status === 'failed' ? 'failed'
                    : 'idle'
                  }
                  onStart={handleStartTraining}
                  onCancel={() => void handleCancelTraining()}
                />
                <ProgressTracker
                  stages={trainingStages}
                  currentStage={job?.stage || ''}
                />
                {viewMode === 'pro' && (
                  <>
                    <TrainingParams
                      params={trainingParams}
                      onParamsChange={setTrainingParams}
                    />
                    <TrainingMonitor
                      data={trainingHistory}
                      currentEpoch={trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].epoch : 0}
                      totalEpochs={trainingParams.epochs}
                    />
                  </>
                )}
                <LogViewer logs={logs} />
              </>
            )}

            {activeView === 'monitor' && (
              <SystemMetrics metrics={metrics} history={metricsHistory} />
            )}

            {activeView === 'export' && (
              <ExportPanel
                format={exportFormat}
                quantization={quantization}
                onFormatChange={setExportFormat}
                onQuantizationChange={setQuantization}
                onExport={() => void handleExport()}
                onOllamaImport={() => void handleOllamaImport()}
                onStartServer={() => void handleStartServer()}
                isExporting={isExporting}
                hasModel={hasGgufModel}
              />
            )}
          </div>
        </main>
      </div>
      <StatusBar metrics={metrics} />
    </div>
  )
}

export default App
