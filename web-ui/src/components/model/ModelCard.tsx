import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Box, Check } from "lucide-react"

interface ModelCardProps {
  id: string
  name: string
  params: string
  ramRequired: number
  vramRequired?: number
  isSelected: boolean
  isAvailable: boolean
  onSelect: (id: string) => void
  className?: string
}

export function ModelCard({
  id,
  name,
  params,
  ramRequired,
  vramRequired,
  isSelected,
  isAvailable,
  onSelect,
  className
}: ModelCardProps) {
  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:border-primary/50",
        isSelected && "border-primary ring-2 ring-primary/20",
        !isAvailable && "opacity-50 cursor-not-allowed",
        className
      )}
      onClick={() => isAvailable && onSelect(id)}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{name}</CardTitle>
          {isSelected && (
            <div className="h-6 w-6 rounded-full bg-primary flex items-center justify-center">
              <Check className="h-4 w-4 text-primary-foreground" />
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Box className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">{params} 参数</span>
          </div>
          <div className="text-xs text-muted-foreground space-y-1">
            <p>RAM: {ramRequired} GB</p>
            {vramRequired && <p>VRAM: {vramRequired} GB</p>}
          </div>
          <div className={cn(
            "text-xs px-2 py-1 rounded-full w-fit",
            isAvailable
              ? "bg-success/10 text-success"
              : "bg-destructive/10 text-destructive"
          )}>
            {isAvailable ? "可用" : "资源不足"}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
