import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Snowflake } from "lucide-react"

interface HeaderProps {
  connected: boolean
  viewMode: 'simple' | 'pro'
  onViewModeChange: (mode: 'simple' | 'pro') => void
  className?: string
}

export function Header({ connected, viewMode, onViewModeChange, className }: HeaderProps) {
  return (
    <header className={cn("border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50", className)}>
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Snowflake className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-xl font-bold tracking-tight">OneClick Distill</h1>
              <p className="text-xs text-muted-foreground">模型蒸馏 · 一键搞定</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            {connected ? (
              <div className="flex items-center gap-2 text-success">
                <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
                <span className="text-sm">已连接</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-destructive">
                <div className="h-2 w-2 rounded-full bg-destructive" />
                <span className="text-sm">离线</span>
              </div>
            )}
          </div>

          <div className="flex items-center bg-secondary rounded-lg p-1">
            <Button
              variant={viewMode === 'simple' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => onViewModeChange('simple')}
              className="h-8"
            >
              简单模式
            </Button>
            <Button
              variant={viewMode === 'pro' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => onViewModeChange('pro')}
              className="h-8"
            >
              专业模式
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
}
