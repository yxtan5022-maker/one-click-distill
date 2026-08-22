export interface Hardware {
  cpu: string
  ram: number
  gpu: string | null
  vram: number | null
  strategy: 'unsloth' | 'transformers' | 'cpu'
  fit: 'green' | 'amber' | 'red'
}

export interface ModelPreset {
  id: string
  name: string
  params: string
  ramRequired: number
  vramRequired?: number
  description: string
}

export interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  progress: number
  status: 'uploading' | 'ready' | 'error'
  path?: string
}

// Mirrors the backend JobState (schema.py) exactly.
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
export type JobStage = 'prepare' | 'data' | 'import' | 'synthetic' | 'train' | 'eval' | 'quantize' | 'done'

export interface JobResult {
  model_dir?: string
  gguf?: string | null
  gguf_url?: string | null
  [key: string]: unknown
}

export interface Job {
  id: string
  status: JobStatus
  stage: JobStage
  progress: number
  message: string
  metrics?: Record<string, unknown>
  error?: string | null
  created_at: number
  updated_at: number
  source: string
  spec?: Record<string, unknown>
  result?: JobResult
}

export interface TrainingParams {
  loraR: number
  loraAlpha: number
  loraDropout: number
  learningRate: number
  epochs: number
  batchSize: number
  gradientAccumulation: number
  maxSeqLength: number
  scheduler: 'cosine' | 'linear' | 'constant'
}

export interface ExportConfig {
  format: 'gguf' | 'onnx' | 'torchscript'
  quantization: 'f16' | 'q8_0' | 'q5_k_m' | 'q4_k_m' | 'q3_k_s'
}

export interface Metrics {
  cpu: number
  ram: number
  gpu?: number
  vram?: number
  timestamp: Date
}

export interface ModelVersion {
  id: string
  name: string
  version: string
  size: number
  format: string
  quantization: string
  createdAt: Date
  metrics?: {
    loss?: number
    accuracy?: number
    rougeL?: number
  }
}

export type ViewMode = 'simple' | 'pro'
export type ActiveView = 'data' | 'model' | 'training' | 'export' | 'evaluate' | 'monitor'
