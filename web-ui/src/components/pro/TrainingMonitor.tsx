import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { Activity, TrendingDown, Zap } from "lucide-react"

interface TrainingDataPoint {
  epoch: number
  step: number
  loss: number
  learningRate: number
  tokensPerSecond?: number
}

interface TrainingMonitorProps {
  data: TrainingDataPoint[]
  currentEpoch: number
  totalEpochs: number
  className?: string
}

export function TrainingMonitor({ data, currentEpoch, totalEpochs, className }: TrainingMonitorProps) {
  const latestData = data[data.length - 1]

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          训练监控
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 bg-secondary rounded-lg">
            <TrendingDown className="h-6 w-6 mx-auto mb-2 text-primary" />
            <div className="text-2xl font-bold">{latestData?.loss.toFixed(4) || '-'}</div>
            <div className="text-xs text-muted-foreground">当前 Loss</div>
          </div>
          <div className="text-center p-4 bg-secondary rounded-lg">
            <Activity className="h-6 w-6 mx-auto mb-2 text-primary" />
            <div className="text-2xl font-bold">{currentEpoch}/{totalEpochs}</div>
            <div className="text-xs text-muted-foreground">训练轮次</div>
          </div>
          <div className="text-center p-4 bg-secondary rounded-lg">
            <Zap className="h-6 w-6 mx-auto mb-2 text-primary" />
            <div className="text-2xl font-bold">{latestData?.tokensPerSecond?.toFixed(0) || '-'}</div>
            <div className="text-xs text-muted-foreground">tokens/sec</div>
          </div>
        </div>

        <div className="h-64">
          {data.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#44403c" />
                <XAxis dataKey="step" stroke="#a8a29e" fontSize={12} />
                <YAxis yAxisId="loss" stroke="#a8a29e" fontSize={12} />
                <YAxis yAxisId="lr" orientation="right" stroke="#a8a29e" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1c1917', border: '1px solid #44403c', borderRadius: '8px' }}
                  labelStyle={{ color: '#fafaf9' }}
                />
                <Legend />
                <Line
                  yAxisId="loss"
                  type="monotone"
                  dataKey="loss"
                  stroke="#c084fc"
                  strokeWidth={2}
                  dot={false}
                  name="Loss"
                />
                <Line
                  yAxisId="lr"
                  type="monotone"
                  dataKey="learningRate"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  name="学习率"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground">
              等待训练数据...
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
