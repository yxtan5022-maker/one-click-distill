import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Database, Eye, Edit3, Trash2, Plus } from "lucide-react"

interface DataItem {
  id: string
  prompt: string
  response: string
}

interface DataManagerProps {
  data: DataItem[]
  onDataAdd: (item: Omit<DataItem, 'id'>) => void
  onDataUpdate: (id: string, item: Partial<DataItem>) => void
  onDataRemove: (id: string) => void
  className?: string
}

export function DataManager({ data, onDataAdd, onDataUpdate, onDataRemove, className }: DataManagerProps) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editPrompt, setEditPrompt] = useState("")
  const [editResponse, setEditResponse] = useState("")
  const [showPreview, setShowPreview] = useState<string | null>(null)

  const startEditing = (item: DataItem) => {
    setEditingId(item.id)
    setEditPrompt(item.prompt)
    setEditResponse(item.response)
  }

  const saveEditing = () => {
    if (editingId) {
      onDataUpdate(editingId, { prompt: editPrompt, response: editResponse })
      setEditingId(null)
    }
  }

  const cancelEditing = () => {
    setEditingId(null)
    setEditPrompt("")
    setEditResponse("")
  }

  const stats = {
    total: data.length,
    avgPromptLength: data.length ? Math.round(data.reduce((sum, d) => sum + d.prompt.length, 0) / data.length) : 0,
    avgResponseLength: data.length ? Math.round(data.reduce((sum, d) => sum + d.response.length, 0) / data.length) : 0,
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            数据管理
          </CardTitle>
          <Button size="sm" onClick={() => onDataAdd({ prompt: "", response: "" })}>
            <Plus className="h-4 w-4 mr-1" />
            添加
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-secondary rounded-lg">
            <div className="text-2xl font-bold">{stats.total}</div>
            <div className="text-xs text-muted-foreground">问答对</div>
          </div>
          <div className="p-3 bg-secondary rounded-lg">
            <div className="text-2xl font-bold">{stats.avgPromptLength}</div>
            <div className="text-xs text-muted-foreground">平均问题长度</div>
          </div>
          <div className="p-3 bg-secondary rounded-lg">
            <div className="text-2xl font-bold">{stats.avgResponseLength}</div>
            <div className="text-xs text-muted-foreground">平均回答长度</div>
          </div>
        </div>

        <div className="space-y-2 max-h-96 overflow-y-auto">
          {data.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              暂无训练数据，请上传文件或手动添加
            </div>
          ) : (
            data.map((item) => (
              <div key={item.id} className="border rounded-lg p-3 space-y-2">
                {editingId === item.id ? (
                  <>
                    <textarea
                      className="w-full h-16 p-2 text-sm bg-background border rounded"
                      value={editPrompt}
                      onChange={(e) => setEditPrompt(e.target.value)}
                      placeholder="问题"
                    />
                    <textarea
                      className="w-full h-16 p-2 text-sm bg-background border rounded"
                      value={editResponse}
                      onChange={(e) => setEditResponse(e.target.value)}
                      placeholder="回答"
                    />
                    <div className="flex gap-2">
                      <Button size="sm" onClick={saveEditing}>保存</Button>
                      <Button size="sm" variant="outline" onClick={cancelEditing}>取消</Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-muted-foreground truncate">{item.prompt}</p>
                        {showPreview === item.id && (
                          <p className="text-sm mt-1">{item.response}</p>
                        )}
                      </div>
                      <div className="flex gap-1">
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-7 w-7"
                          onClick={() => setShowPreview(showPreview === item.id ? null : item.id)}
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-7 w-7"
                          onClick={() => startEditing(item)}
                        >
                          <Edit3 className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-7 w-7"
                          onClick={() => onDataRemove(item.id)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                    {showPreview !== item.id && (
                      <p className="text-xs text-muted-foreground truncate">{item.response}</p>
                    )}
                  </>
                )}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}
