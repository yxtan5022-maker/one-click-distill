import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Download, Server, ExternalLink } from "lucide-react"

interface ExportPanelProps {
  format: string
  quantization: string
  onFormatChange: (format: string) => void
  onQuantizationChange: (quantization: string) => void
  onExport: () => void
  onOllamaImport: () => void
  onStartServer: () => void
  isExporting: boolean
  hasModel: boolean
  className?: string
}

export function ExportPanel({
  format,
  quantization,
  onFormatChange,
  onQuantizationChange,
  onExport,
  onOllamaImport,
  onStartServer,
  isExporting,
  hasModel,
  className
}: ExportPanelProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Download className="h-5 w-5 text-primary" />
          导出
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">导出格式</label>
            <Select value={format} onValueChange={onFormatChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gguf">GGUF</SelectItem>
                <SelectItem value="onnx">ONNX</SelectItem>
                <SelectItem value="torchscript">TorchScript</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">量化格式</label>
            <Select value={quantization} onValueChange={onQuantizationChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="f16">F16 (无损)</SelectItem>
                <SelectItem value="q8_0">Q8_0 (高精度)</SelectItem>
                <SelectItem value="q5_k_m">Q5_K_M (平衡)</SelectItem>
                <SelectItem value="q4_k_m">Q4_K_M (推荐)</SelectItem>
                <SelectItem value="q3_k_s">Q3_K_S (小体积)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex gap-3">
          <Button
            onClick={onExport}
            disabled={!hasModel || isExporting}
            className="flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            {isExporting ? '导出中...' : '导出 GGUF'}
          </Button>
          <Button
            variant="outline"
            onClick={onOllamaImport}
            disabled={!hasModel}
            className="flex items-center gap-2"
          >
            <ExternalLink className="h-4 w-4" />
            导入 Ollama
          </Button>
          <Button
            variant="outline"
            onClick={onStartServer}
            disabled={!hasModel}
            className="flex items-center gap-2"
          >
            <Server className="h-4 w-4" />
            启动 API
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
