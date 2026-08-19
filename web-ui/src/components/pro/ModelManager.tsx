import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Boxes, Trash2, Download, Copy, BarChart3 } from "lucide-react"
import type { ModelVersion } from "@/types"

interface ModelManagerProps {
  models: ModelVersion[]
  selectedModel: string | null
  onSelectModel: (id: string) => void
  onDeleteModel: (id: string) => void
  onCompareModels: (ids: string[]) => void
  className?: string
}

export function ModelManager({
  models,
  selectedModel,
  onSelectModel,
  onDeleteModel,
  onCompareModels,
  className
}: ModelManagerProps) {
  const [compareMode, setCompareMode] = useState(false)
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([])

  const toggleCompareSelection = (id: string) => {
    setSelectedForCompare(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    )
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Boxes className="h-5 w-5 text-primary" />
            模型管理
          </CardTitle>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant={compareMode ? 'default' : 'outline'}
              onClick={() => {
                setCompareMode(!compareMode)
                setSelectedForCompare([])
              }}
            >
              <BarChart3 className="h-4 w-4 mr-1" />
              对比
            </Button>
            {compareMode && selectedForCompare.length >= 2 && (
              <Button
                size="sm"
                onClick={() => onCompareModels(selectedForCompare)}
              >
                对比 {selectedForCompare.length} 个模型
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {models.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            暂无训练模型
          </div>
        ) : (
          <div className="space-y-2">
            {models.map((model) => (
              <div
                key={model.id}
                className={`p-4 border rounded-lg transition-colors ${
                  selectedModel === model.id ? 'border-primary bg-primary/5' : 'hover:border-primary/30'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {compareMode && (
                      <input
                        type="checkbox"
                        checked={selectedForCompare.includes(model.id)}
                        onChange={() => toggleCompareSelection(model.id)}
                        className="h-4 w-4"
                      />
                    )}
                    <div
                      className="cursor-pointer flex-1"
                      onClick={() => onSelectModel(model.id)}
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{model.name}</span>
                        <span className="text-xs px-2 py-0.5 bg-secondary rounded">{model.version}</span>
                        <span className="text-xs text-muted-foreground">{model.format}</span>
                      </div>
                      <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                        <span>大小: {formatSize(model.size)}</span>
                        <span>量化: {model.quantization}</span>
                        <span>{new Date(model.createdAt).toLocaleDateString()}</span>
                      </div>
                      {model.metrics && (
                        <div className="flex items-center gap-4 mt-1 text-xs">
                          {model.metrics.loss !== undefined && (
                            <span>Loss: {model.metrics.loss.toFixed(4)}</span>
                          )}
                          {model.metrics.rougeL !== undefined && (
                            <span>ROUGE-L: {(model.metrics.rougeL * 100).toFixed(1)}%</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-1">
                    <Button size="icon" variant="ghost" className="h-8 w-8">
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button size="icon" variant="ghost" className="h-8 w-8">
                      <Copy className="h-4 w-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-8 w-8"
                      onClick={() => onDeleteModel(model.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
