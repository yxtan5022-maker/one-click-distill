import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Settings2, Info } from "lucide-react"
import type { TrainingParams as TrainingParamsType } from "@/types"

interface TrainingParamsProps {
  params: TrainingParamsType
  onParamsChange: (params: TrainingParamsType) => void
  className?: string
}

function InfoTooltip({ text }: { text: string }) {
  return (
    <span className="relative group">
      <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 text-xs bg-popover text-popover-foreground rounded-lg shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none w-48 z-50">
        {text}
      </span>
    </span>
  )
}

export function TrainingParams({ params, onParamsChange, className }: TrainingParamsProps) {
  const handleChange = (key: keyof TrainingParamsType, value: number | string) => {
    onParamsChange({ ...params, [key]: value })
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Settings2 className="h-5 w-5 text-primary" />
          训练参数
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">LoRA Rank (r)</label>
              <InfoTooltip text="LoRA低秩分解的秩，越大表达能力越强但消耗更多显存" />
            </div>
            <Input
              type="number"
              min={1}
              max={64}
              value={params.loraR}
              onChange={(e) => handleChange('loraR', parseInt(e.target.value) || 1)}
            />
            <p className="text-xs text-muted-foreground">推荐: 8-16</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">LoRA Alpha</label>
              <InfoTooltip text="LoRA缩放系数，通常设为rank的2倍" />
            </div>
            <Input
              type="number"
              min={1}
              max={128}
              value={params.loraAlpha}
              onChange={(e) => handleChange('loraAlpha', parseInt(e.target.value) || 1)}
            />
            <p className="text-xs text-muted-foreground">推荐: rank × 2</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">LoRA Dropout</label>
              <InfoTooltip text="防止过拟合的随机丢弃率" />
            </div>
            <Input
              type="number"
              min={0}
              max={0.5}
              step={0.05}
              value={params.loraDropout}
              onChange={(e) => handleChange('loraDropout', parseFloat(e.target.value) || 0)}
            />
            <p className="text-xs text-muted-foreground">推荐: 0.05-0.1</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">学习率</label>
              <InfoTooltip text="每步参数更新的步长" />
            </div>
            <Input
              type="number"
              min={0.000001}
              max={0.01}
              step={0.000001}
              value={params.learningRate}
              onChange={(e) => handleChange('learningRate', parseFloat(e.target.value) || 0.0002)}
            />
            <p className="text-xs text-muted-foreground">推荐: 1e-4 ~ 5e-4</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">训练轮次</label>
              <InfoTooltip text="完整遍历数据集的次数" />
            </div>
            <Input
              type="number"
              min={1}
              max={100}
              value={params.epochs}
              onChange={(e) => handleChange('epochs', parseInt(e.target.value) || 3)}
            />
            <p className="text-xs text-muted-foreground">推荐: 3-10</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">批次大小</label>
              <InfoTooltip text="每次送入模型的样本数" />
            </div>
            <Input
              type="number"
              min={1}
              max={64}
              value={params.batchSize}
              onChange={(e) => handleChange('batchSize', parseInt(e.target.value) || 4)}
            />
            <p className="text-xs text-muted-foreground">显存不足时减小</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">梯度累积步数</label>
              <InfoTooltip text="模拟更大批次的梯度累积" />
            </div>
            <Input
              type="number"
              min={1}
              max={32}
              value={params.gradientAccumulation}
              onChange={(e) => handleChange('gradientAccumulation', parseInt(e.target.value) || 4)}
            />
            <p className="text-xs text-muted-foreground">有效批次 = batch × accumulation</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">最大序列长度</label>
              <InfoTooltip text="输入文本截断的最大token数" />
            </div>
            <Input
              type="number"
              min={64}
              max={4096}
              step={64}
              value={params.maxSeqLength}
              onChange={(e) => handleChange('maxSeqLength', parseInt(e.target.value) || 512)}
            />
            <p className="text-xs text-muted-foreground">更长占用更多显存</p>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">学习率调度器</label>
            <InfoTooltip text="训练过程中学习率的变化策略" />
          </div>
          <Select value={params.scheduler} onValueChange={(v) => handleChange('scheduler', v)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cosine">Cosine (余弦退火，推荐)</SelectItem>
              <SelectItem value="linear">Linear (线性衰减)</SelectItem>
              <SelectItem value="constant">Constant (恒定学习率)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  )
}
