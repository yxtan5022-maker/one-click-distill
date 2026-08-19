import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2, Circle, Loader2 } from "lucide-react"

interface ProgressTrackerProps {
  stages: Array<{
    id: string
    name: string
    status: 'pending' | 'running' | 'completed' | 'failed'
    progress?: number
  }>
  currentStage: string
  className?: string
}

export function ProgressTracker({ stages, currentStage: _currentStage, className }: ProgressTrackerProps) {
  const getStageIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-5 w-5 text-success" />
      case 'running':
        return <Loader2 className="h-5 w-5 text-primary animate-spin" />
      case 'failed':
        return <div className="h-5 w-5 rounded-full bg-destructive" />
      default:
        return <Circle className="h-5 w-5 text-muted-foreground" />
    }
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg">训练进度</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {stages.map((stage, index) => (
            <div key={stage.id} className="flex items-center gap-4">
              <div className="flex-shrink-0">{getStageIcon(stage.status)}</div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className={cn(
                    "text-sm font-medium",
                    stage.status === 'running' && "text-primary",
                    stage.status === 'completed' && "text-success",
                    stage.status === 'failed' && "text-destructive"
                  )}>
                    {stage.name}
                  </span>
                  {stage.progress !== undefined && (
                    <span className="text-xs text-muted-foreground">
                      {stage.progress}%
                    </span>
                  )}
                </div>
                {stage.progress !== undefined && (
                  <div className="mt-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all",
                        stage.status === 'completed' && "bg-success",
                        stage.status === 'running' && "bg-primary",
                        stage.status === 'failed' && "bg-destructive"
                      )}
                      style={{ width: `${stage.progress}%` }}
                    />
                  </div>
                )}
              </div>
              {index < stages.length - 1 && (
                <div className="absolute left-[18px] top-[40px] w-0.5 h-4 bg-border" />
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
