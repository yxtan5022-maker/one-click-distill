import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Play, Pause, Square, Loader2 } from "lucide-react"

interface TrainingControlProps {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed'
  onStart: () => void
  onPause: () => void
  onResume: () => void
  onCancel: () => void
  className?: string
}

export function TrainingControl({
  status,
  onStart,
  onPause,
  onResume,
  onCancel,
  className
}: TrainingControlProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Play className="h-5 w-5 text-primary" />
          训练控制
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-3">
          {status === 'idle' && (
            <Button onClick={onStart} className="flex items-center gap-2">
              <Play className="h-4 w-4" />
              开始训练
            </Button>
          )}
          {status === 'running' && (
            <>
              <Button variant="outline" onClick={onPause} className="flex items-center gap-2">
                <Pause className="h-4 w-4" />
                暂停
              </Button>
              <Button variant="destructive" onClick={onCancel} className="flex items-center gap-2">
                <Square className="h-4 w-4" />
                取消
              </Button>
              <div className="flex items-center gap-2 text-primary">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">训练中...</span>
              </div>
            </>
          )}
          {status === 'paused' && (
            <>
              <Button onClick={onResume} className="flex items-center gap-2">
                <Play className="h-4 w-4" />
                继续
              </Button>
              <Button variant="destructive" onClick={onCancel} className="flex items-center gap-2">
                <Square className="h-4 w-4" />
                取消
              </Button>
            </>
          )}
          {status === 'completed' && (
            <div className="flex items-center gap-2 text-success">
              <span className="text-sm font-medium">训练完成</span>
            </div>
          )}
          {status === 'failed' && (
            <div className="flex items-center gap-2 text-destructive">
              <span className="text-sm font-medium">训练失败</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
