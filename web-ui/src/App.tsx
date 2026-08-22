import { useState, useEffect, useCallback } from "react"
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
import { modelPresets } from "@/components/model/ModelPresets"
import type { UploadedFile, Metrics, ViewMode, ActiveView, Job, TrainingParams as TrainingParamsType, ModelVersion } from "@/types"

const API_BASE = "http://127.0.0.1:8080"

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

const sampleModels: ModelVersion[] = [
  { id: '1', name: 'Qwen2.5-3B-Distilled', version: 'v1.0', size: 1800000000, format: 'GGUF', quantization: 'Q4_K_M', createdAt: new Date('2026-08-18'), metrics: { loss: 0.2345, rougeL: 0.82 } },
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
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>(sampleModels)

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

  const handleFilesAdd = useCallback((newFiles: File[]) => {
    const uploadedFiles: UploadedFile[] = newFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: 'ready'
    }))
    setFiles(prev => [...prev, ...uploadedFiles])
  }, [])

  const handleFileRemove = useCallback((id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id))
  }, [])

  const handleStartTraining = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          files: files.map(f => f.name),
          model: selectedModel,
          teacher: {
            provider: teacherProvider,
            model: teacherModel,
            apiKey
          },
          trainingParams
        })
      })
      if (response.ok) {
        const data = await response.json()
        setJob(data)
        setLogs(prev => [...prev, `[INFO] 训练任务已启动: ${data.id}`])
      }
    } catch (error) {
      setLogs(prev => [...prev, `[ERROR] 启动训练失败: ${error}`])
    }
  }, [files, selectedModel, teacherProvider, teacherModel, apiKey, trainingParams])

  const handlePauseTraining = useCallback(() => {
    setLogs(prev => [...prev, "[INFO] 训练已暂停"])
  }, [])

  const handleResumeTraining = useCallback(() => {
    setLogs(prev => [...prev, "[INFO] 训练已继续"])
  }, [])

  const handleCancelTraining = useCallback(() => {
    setJob(null)
    setLogs(prev => [...prev, "[INFO] 训练已取消"])
  }, [])

  const handleExport = useCallback(async () => {
    setIsExporting(true)
    try {
      const response = await fetch(`${API_BASE}/api/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: exportFormat, quantization })
      })
      if (response.ok) {
        setLogs(prev => [...prev, "[SUCCESS] 导出完成"])
      }
    } catch (error) {
      setLogs(prev => [...prev, `[ERROR] 导出失败: ${error}`])
    } finally {
      setIsExporting(false)
    }
  }, [exportFormat, quantization])

  const handleOllamaImport = useCallback(() => {
    setLogs(prev => [...prev, "[INFO] 正在导入到 Ollama..."])
  }, [])

  const handleStartServer = useCallback(() => {
    setLogs(prev => [...prev, "[INFO] 正在启动本地 API 服务器..."])
  }, [])

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

  const handleOneClickPipeline = useCallback(async (config: {
    model: string
    params: TrainingParamsType
    autoMode: boolean
  }) => {
    setLogs(prev => [...prev, `[INFO] 启动一键蒸馏: 模型=${config.model}, 自动模式=${config.autoMode}`])
    setTrainingParams(config.params)
    setSelectedModel(config.model)
    await handleStartTraining()
  }, [handleStartTraining])

  const trainingStages: Array<{
    id: string
    name: string
    status: 'pending' | 'running' | 'completed' | 'failed'
    progress?: number
  }> = [
    { id: 'data', name: '数据处理', status: job?.stage === 'data' ? 'running' : job ? 'completed' : 'pending', progress: job?.stage === 'data' ? job.progress : undefined },
    { id: 'training', name: '模型训练', status: job?.stage === 'training' ? 'running' : job?.stage === 'data' ? 'pending' : 'pending', progress: job?.stage === 'training' ? job.progress : undefined },
    { id: 'quantizing', name: '模型量化', status: job?.stage === 'quantizing' ? 'running' : 'pending', progress: job?.stage === 'quantizing' ? job.progress : undefined },
    { id: 'exporting', name: '导出模型', status: job?.stage === 'exporting' ? 'running' : 'pending', progress: job?.stage === 'exporting' ? job.progress : undefined },
  ]

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
                  isRunning={job?.status === 'running'}
                />
                <TrainingControl
                  status={job?.status === 'running' ? 'running' : job?.status === 'completed' ? 'completed' : job?.status === 'failed' ? 'failed' : 'idle'}
                  onStart={handleStartTraining}
                  onPause={handlePauseTraining}
                  onResume={handleResumeTraining}
                  onCancel={handleCancelTraining}
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
                onExport={handleExport}
                onOllamaImport={handleOllamaImport}
                onStartServer={handleStartServer}
                isExporting={isExporting}
                hasModel={!!selectedModel}
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
