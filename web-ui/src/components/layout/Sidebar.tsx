import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Database, Box, Settings2, Download, BarChart3, Activity } from "lucide-react"

interface SidebarProps {
  activeView: string
  onViewChange: (view: string) => void
  className?: string
}

const navItems = [
  { id: 'data', label: '数据', icon: Database },
  { id: 'model', label: '模型', icon: Box },
  { id: 'training', label: '训练', icon: Settings2 },
  { id: 'export', label: '导出', icon: Download },
  { id: 'monitor', label: '监控', icon: Activity },
  { id: 'evaluate', label: '评估', icon: BarChart3 },
]

export function Sidebar({ activeView, onViewChange, className }: SidebarProps) {
  return (
    <aside className={cn("w-16 bg-card border-r flex flex-col items-center py-4 gap-2", className)}>
      {navItems.map((item) => (
        <Button
          key={item.id}
          variant={activeView === item.id ? 'default' : 'ghost'}
          size="icon"
          onClick={() => onViewChange(item.id)}
          className="w-12 h-12 flex flex-col gap-1"
          title={item.label}
        >
          <item.icon className="h-5 w-5" />
          <span className="text-[10px] leading-none">{item.label}</span>
        </Button>
      ))}
    </aside>
  )
}
