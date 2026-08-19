import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ModelCard } from "./ModelCard"
import { Box } from "lucide-react"
import type { ModelPreset } from "@/types"

interface ModelGridProps {
  models: ModelPreset[]
  selectedModel: string | null
  availableRam: number
  availableVram: number | null
  onSelectModel: (id: string) => void
  className?: string
}

const defaultModels: ModelPreset[] = [
  { id: '0.5b', name: 'Qwen2.5 0.5B', params: '0.5B', ramRequired: 2, vramRequired: 1, description: '最小模型，适合快速测试' },
  { id: '1.5b', name: 'Qwen2.5 1.5B', params: '1.5B', ramRequired: 4, vramRequired: 2, description: '轻量级模型，适合简单任务' },
  { id: '3b', name: 'Qwen2.5 3B', params: '3B', ramRequired: 8, vramRequired: 4, description: '平衡性能和资源' },
  { id: '7b', name: 'Qwen2.5 7B', params: '7B', ramRequired: 16, vramRequired: 8, description: '高性能模型' },
  { id: '13b', name: 'Qwen2.5 13B', params: '13B', ramRequired: 32, vramRequired: 16, description: '大型模型，需要更多资源' },
  { id: '30b', name: 'Qwen2.5 30B', params: '30B', ramRequired: 64, vramRequired: 24, description: '最大模型，需要高端硬件' },
]

export function ModelGrid({
  models = defaultModels,
  selectedModel,
  availableRam,
  availableVram,
  onSelectModel,
  className
}: ModelGridProps) {
  const checkAvailability = (model: ModelPreset) => {
    const ramOk = availableRam >= model.ramRequired
    const vramOk = model.vramRequired ? (availableVram ?? 0) >= model.vramRequired : true
    return ramOk && vramOk
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Box className="h-5 w-5 text-primary" />
          模型选择
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              {...model}
              isSelected={selectedModel === model.id}
              isAvailable={checkAvailability(model)}
              onSelect={onSelectModel}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
