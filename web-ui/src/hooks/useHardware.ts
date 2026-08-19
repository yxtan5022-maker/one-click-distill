import { useState, useEffect } from "react"
import type { Hardware } from "@/types"

const API_BASE = "http://127.0.0.1:8080"

export function useHardware() {
  const [hardware, setHardware] = useState<Hardware | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchHardware = async () => {
      try {
        setLoading(true)
        const response = await fetch(`${API_BASE}/api/hardware`)
        if (response.ok) {
          const data = await response.json()
          setHardware({
            cpu: data.device_name || 'CPU',
            ram: data.total_ram_gb || 0,
            gpu: data.device_name || null,
            vram: data.total_vram_gb || null,
            strategy: data.strategy?.backend || 'cpu',
            fit: data.strategy?.backend === 'unsloth' ? 'green' :
                 data.strategy?.backend === 'transformers' ? 'amber' : 'red'
          })
          setError(null)
        } else {
          setError('Failed to fetch hardware info')
        }
      } catch (err) {
        setError('Cannot connect to backend')
        // Fallback to mock data for development
        setHardware({
          cpu: 'Intel Core i7-12700K',
          ram: 16,
          gpu: null,
          vram: null,
          strategy: 'cpu',
          fit: 'amber'
        })
      } finally {
        setLoading(false)
      }
    }

    fetchHardware()
    const interval = setInterval(fetchHardware, 30000)
    return () => clearInterval(interval)
  }, [])

  return { hardware, loading, error }
}
