import type { ModelPreset } from "@/types"

export const modelPresets: ModelPreset[] = [
  {
    id: 'qwen2.5-0.5b',
    name: 'Qwen2.5 0.5B',
    params: '0.5B',
    ramRequired: 2,
    vramRequired: 1,
    description: '最小模型，适合快速测试和资源受限环境'
  },
  {
    id: 'qwen2.5-1.5b',
    name: 'Qwen2.5 1.5B',
    params: '1.5B',
    ramRequired: 4,
    vramRequired: 2,
    description: '轻量级模型，适合简单任务和移动设备'
  },
  {
    id: 'qwen2.5-3b',
    name: 'Qwen2.5 3B',
    params: '3B',
    ramRequired: 8,
    vramRequired: 4,
    description: '平衡性能和资源，适合大多数场景'
  },
  {
    id: 'qwen2.5-7b',
    name: 'Qwen2.5 7B',
    params: '7B',
    ramRequired: 16,
    vramRequired: 8,
    description: '高性能模型，适合复杂任务'
  },
  {
    id: 'qwen2.5-14b',
    name: 'Qwen2.5 14B',
    params: '14B',
    ramRequired: 32,
    vramRequired: 16,
    description: '大型模型，需要较多资源'
  },
  {
    id: 'qwen2.5-32b',
    name: 'Qwen2.5 32B',
    params: '32B',
    ramRequired: 64,
    vramRequired: 24,
    description: '超大模型，需要高端硬件'
  },
  {
    id: 'llama3.1-8b',
    name: 'Llama 3.1 8B',
    params: '8B',
    ramRequired: 16,
    vramRequired: 8,
    description: 'Meta开源模型，多语言能力强'
  },
  {
    id: 'mistral-7b',
    name: 'Mistral 7B',
    params: '7B',
    ramRequired: 16,
    vramRequired: 8,
    description: '高效模型，推理速度快'
  },
  {
    id: 'yi-1.5-6b',
    name: 'Yi 1.5 6B',
    params: '6B',
    ramRequired: 12,
    vramRequired: 6,
    description: '零一万物模型，中文优化'
  },
  {
    id: 'phi-3-mini',
    name: 'Phi-3 Mini',
    params: '3.8B',
    ramRequired: 8,
    vramRequired: 4,
    description: '微软小模型，性能超预期'
  },
  {
    id: 'gemma-2-2b',
    name: 'Gemma 2 2B',
    params: '2B',
    ramRequired: 4,
    vramRequired: 2,
    description: 'Google轻量模型，适合边缘部署'
  },
  {
    id: 'deepseek-llm-7b',
    name: 'DeepSeek LLM 7B',
    params: '7B',
    ramRequired: 16,
    vramRequired: 8,
    description: 'DeepSeek开源模型，代码能力强'
  }
]

export const quantizationFormats = [
  { id: 'f16', name: 'F16', description: '无损，体积最大', sizeMultiplier: 1.0 },
  { id: 'q8_0', name: 'Q8_0', description: '高精度，体积较大', sizeMultiplier: 0.5 },
  { id: 'q5_k_m', name: 'Q5_K_M', description: '平衡精度和体积', sizeMultiplier: 0.35 },
  { id: 'q4_k_m', name: 'Q4_K_M', description: '推荐，最佳平衡', sizeMultiplier: 0.25 },
  { id: 'q4_k_s', name: 'Q4_K_S', description: '更小体积', sizeMultiplier: 0.22 },
  { id: 'q3_k_m', name: 'Q3_K_M', description: '小体积，精度损失', sizeMultiplier: 0.2 },
  { id: 'q3_k_s', name: 'Q3_K_S', description: '最小体积', sizeMultiplier: 0.18 },
  { id: 'q2_k', name: 'Q2_K', description: '极小体积，精度损失大', sizeMultiplier: 0.15 },
]

export function getModelById(id: string): ModelPreset | undefined {
  return modelPresets.find(m => m.id === id)
}

export function getCompatibleModels(ram: number, vram: number | null): ModelPreset[] {
  return modelPresets.filter(model => {
    const ramOk = ram >= model.ramRequired
    const vramOk = model.vramRequired ? (vram ?? 0) >= model.vramRequired : true
    return ramOk && vramOk
  })
}

export function getRecommendedModel(ram: number, vram: number | null): ModelPreset {
  const compatible = getCompatibleModels(ram, vram)
  if (compatible.length === 0) return modelPresets[0]
  return compatible[compatible.length - 1]
}
