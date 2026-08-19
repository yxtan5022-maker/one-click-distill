import { cn } from "@/lib/utils"
import { Cpu, HardDrive, Thermometer } from "lucide-react"
import type { Metrics } from "@/types"

interface StatusBarProps {
  metrics: Metrics | null
  className?: string
}

export function StatusBar({ metrics, className }: StatusBarProps) {
  if (!metrics) return null

  return (
    <footer className={cn("border-t bg-card/50 backdrop-blur-sm h-8 flex items-center px-4 text-xs text-muted-foreground", className)}>
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Cpu className="h-3 w-3" />
            <span>CPU: {metrics.cpu.toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-2">
            <HardDrive className="h-3 w-3" />
            <span>RAM: {metrics.ram.toFixed(1)}%</span>
          </div>
          {metrics.gpu !== undefined && (
            <div className="flex items-center gap-2">
              <Thermometer className="h-3 w-3" />
              <span>GPU: {metrics.gpu.toFixed(1)}%</span>
            </div>
          )}
        </div>
        <div className="text-muted-foreground/60">
          OneClick Distill v0.1.0
        </div>
      </div>
    </footer>
  )
}
