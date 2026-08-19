import { useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollText } from "lucide-react"

interface LogViewerProps {
  logs: string[]
  className?: string
}

export function LogViewer({ logs, className }: LogViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  const formatLog = (log: string) => {
    if (log.includes('ERROR') || log.includes('error')) {
      return <span className="text-destructive">{log}</span>
    }
    if (log.includes('WARNING') || log.includes('warn')) {
      return <span className="text-warning">{log}</span>
    }
    if (log.includes('INFO') || log.includes('info')) {
      return <span className="text-muted-foreground">{log}</span>
    }
    if (log.includes('SUCCESS') || log.includes('success')) {
      return <span className="text-success">{log}</span>
    }
    return log
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <ScrollText className="h-5 w-5 text-primary" />
          训练日志
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          ref={scrollRef}
          className="h-64 overflow-y-auto bg-background rounded-lg p-4 font-mono text-xs space-y-1"
        >
          {logs.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">
              暂无日志
            </p>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="flex gap-2">
                <span className="text-muted-foreground/50 select-none">
                  {String(index + 1).padStart(4, '0')}
                </span>
                <span>{formatLog(log)}</span>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}
