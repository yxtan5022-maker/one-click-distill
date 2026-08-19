import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Cpu, HardDrive, Thermometer, Activity } from "lucide-react"
import type { Metrics } from "@/types"

interface SystemMetricsProps {
  metrics: Metrics | null
  history: Metrics[]
  className?: string
}

export function SystemMetrics({ metrics, history: _history, className }: SystemMetricsProps) {
  if (!metrics) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            系统监控
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            等待数据...
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          系统监控
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 bg-secondary rounded-lg">
            <Cpu className="h-6 w-6 mx-auto mb-2 text-primary" />
            <div className="text-2xl font-bold">{metrics.cpu.toFixed(1)}%</div>
            <div className="text-xs text-muted-foreground">CPU</div>
          </div>
          <div className="text-center p-4 bg-secondary rounded-lg">
            <HardDrive className="h-6 w-6 mx-auto mb-2 text-primary" />
            <div className="text-2xl font-bold">{metrics.ram.toFixed(1)}%</div>
            <div className="text-xs text-muted-foreground">RAM</div>
          </div>
          {metrics.gpu !== undefined && (
            <div className="text-center p-4 bg-secondary rounded-lg">
              <Thermometer className="h-6 w-6 mx-auto mb-2 text-primary" />
              <div className="text-2xl font-bold">{metrics.gpu.toFixed(1)}%</div>
              <div className="text-xs text-muted-foreground">GPU</div>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span>CPU 使用率</span>
              <span>{metrics.cpu.toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all"
                style={{ width: `${metrics.cpu}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span>RAM 使用率</span>
              <span>{metrics.ram.toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all"
                style={{ width: `${metrics.ram}%` }}
              />
            </div>
          </div>
          {metrics.gpu !== undefined && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>GPU 使用率</span>
                <span>{metrics.gpu.toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all"
                  style={{ width: `${metrics.gpu}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
