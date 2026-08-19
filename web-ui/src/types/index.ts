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
}

export interface Job {
  id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  stage: 'data' | 'training' | 'quantizing' | 'exporting' | 'done'
  progress: number
  logs: string[]
  createdAt: Date
  completedAt?: Date
  model?: string
  dataset?: string
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
