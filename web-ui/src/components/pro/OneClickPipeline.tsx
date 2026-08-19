import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Zap, CheckCircle2, AlertTriangle, Loader2, ChevronRight } from "lucide-react"
import type { Hardware, TrainingParams } from "@/types"

interface OneClickPipelineProps {
  hardware: Hardware | null
  onStartPipeline: (config: {
    model: string
    params: TrainingParams
    autoMode: boolean
  }) => void
  isRunning: boolean
  className?: string
}

interface PipelineStep {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'error'
  message?: string
}

const defaultSteps: PipelineStep[] = [
  { id: 'detect', name: '硬件检测', status: 'pending' },
  { id: 'select', name: '模型选择', status: 'pending' },
  { id: 'config', name: '参数配置', status: 'pending' },
  { id: 'ready', name: '准备就绪', status: 'pending' },
]

export function OneClickPipeline({ hardware, onStartPipeline, isRunning, className }: OneClickPipelineProps) {
  const [steps, setSteps] = useState<PipelineStep[]>(defaultSteps)
  const [autoMode, setAutoMode] = useState(true)

  useEffect(() => {
    if (hardware) {
      setSteps(prev => prev.map(s =>
        s.id === 'detect' ? { ...s, status: 'completed', message: `${hardware.strategy} 模式` } : s
      ))
    }
  }, [hardware])

  const getRecommendedModel = () => {
    if (!hardware) return '0.5b'
    const totalMemory = (hardware.ram || 0) + (hardware.vram || 0)
    if (totalMemory >= 24) return '7b'
    if (totalMemory >= 12) return '3b'
    if (totalMemory >= 6) return '1.5b'
    return '0.5b'
  }

  const getRecommendedParams = (): TrainingParams => {
    if (!hardware) {
      return {
        loraR: 8, loraAlpha: 16, loraDropout: 0.05,
        learningRate: 0.0002, epochs: 3, batchSize: 1,
        gradientAccumulation: 8, maxSeqLength: 512, scheduler: 'cosine'
      }
    }

    if (hardware.strategy === 'unsloth') {
      return {
        loraR: 16, loraAlpha: 32, loraDropout: 0.05,
        learningRate: 0.0002, epochs: 3, batchSize: 2,
        gradientAccumulation: 4, maxSeqLength: 1024, scheduler: 'cosine'
      }
    }

    return {
      loraR: 8, loraAlpha: 16, loraDropout: 0.1,
      learningRate: 0.0001, epochs: 3, batchSize: 1,
      gradientAccumulation: 8, maxSeqLength: 512, scheduler: 'cosine'
    }
  }

  const handleStart = () => {
    const config = {
      model: getRecommendedModel(),
      params: getRecommendedParams(),
      autoMode
    }
    onStartPipeline(config)
  }

  const recommendedModel = getRecommendedModel()
  const recommendedParams = getRecommendedParams()

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Zap className="h-5 w-5 text-primary" />
          一键蒸馏
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-3">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center gap-3">
              <div className="flex-shrink-0">
                {step.status === 'completed' && (
                  <CheckCircle2 className="h-5 w-5 text-success" />
                )}
                {step.status === 'running' && (
                  <Loader2 className="h-5 w-5 text-primary animate-spin" />
                )}
                {step.status === 'error' && (
                  <AlertTriangle className="h-5 w-5 text-destructive" />
                )}
                {step.status === 'pending' && (
                  <div className="h-5 w-5 rounded-full border-2 border-muted-foreground/30" />
                )}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">{step.name}</p>
                {step.message && (
                  <p className="text-xs text-muted-foreground">{step.message}</p>
                )}
              </div>
              {index < steps.length - 1 && (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
            </div>
          ))}
        </div>

        {hardware && (
          <div className="p-4 bg-secondary rounded-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">自动配置</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoMode}
                  onChange={(e) => setAutoMode(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-muted rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div>
              </label>
            </div>
            <div className="text-xs text-muted-foreground space-y-1">
              <p>推荐模型: <span className="text-foreground font-medium">{recommendedModel.toUpperCase()}</span></p>
              <p>训练后端: <span className="text-foreground font-medium">{hardware.strategy}</span></p>
              <p>序列长度: <span className="text-foreground font-medium">{recommendedParams.maxSeqLength}</span></p>
              <p>批次大小: <span className="text-foreground font-medium">{recommendedParams.batchSize}</span></p>
            </div>
          </div>
        )}

        <Button
          onClick={handleStart}
          disabled={!hardware || isRunning}
          className="w-full"
          size="lg"
        >
          {isRunning ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              正在蒸馏...
            </>
          ) : (
            <>
              <Zap className="h-4 w-4 mr-2" />
              一键开始蒸馏
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  )
}
