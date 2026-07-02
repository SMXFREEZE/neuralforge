'use client'
import { useEffect, useRef } from 'react'

interface Props {
  theme: 'dark' | 'light'
}

declare global {
  interface Window {
    VANTA?: {
      FOG: (opts: Record<string, unknown>) => { destroy: () => void }
      CLOUDS: (opts: Record<string, unknown>) => { destroy: () => void }
    }
  }
}

export default function VantaBg({ theme }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const effectRef = useRef<{ destroy: () => void } | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const init = () => {
      if (cancelled || !ref.current || !window.VANTA) {
        timer = setTimeout(init, 80)
        return
      }
      if (effectRef.current) {
        effectRef.current.destroy()
        effectRef.current = null
      }
      try {
        if (theme === 'light') {
          effectRef.current = window.VANTA.CLOUDS({
            el: ref.current,
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200,
            minWidth: 200,
            skyColor: 0xc2daf0,
            cloudColor: 0xe4f0fb,
            cloudShadowColor: 0x93b8d6,
            sunColor: 0xedb96a,
            sunGlareColor: 0xf8d898,
            sunlightColor: 0xfef4d8,
            speed: 0.7,
          })
        } else {
          effectRef.current = window.VANTA.FOG({
            el: ref.current,
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200,
            minWidth: 200,
            highlightColor: 0x111830,
            midtoneColor: 0x09101f,
            lowlightColor: 0x07090b,
            baseColor: 0x07090b,
            blurFactor: 0.65,
            speed: 0.6,
            zoom: 0.9,
          })
        }
      } catch {
        timer = setTimeout(init, 200)
      }
    }

    init()
    return () => {
      cancelled = true
      clearTimeout(timer)
      if (effectRef.current) {
        effectRef.current.destroy()
        effectRef.current = null
      }
    }
  }, [theme])

  return (
    <div
      ref={ref}
      aria-hidden="true"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: -1,
        pointerEvents: 'none',
      }}
    />
  )
}
