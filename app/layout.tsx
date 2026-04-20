import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import Script from 'next/script'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700', '800'],
  variable: '--font-inter',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'NeuralForge — FPGA Neural Network Accelerator',
  description: 'INT8 LeNet-5 inference accelerator dashboard. Draw digits, explore the 4x4 systolic array, and get AI-powered analysis.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
      </head>
      <body className={`${inter.variable} ${jetbrainsMono.variable}`}>
        {children}
        {/* TF.js for client-side MNIST classification */}
        <Script
          src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"
          strategy="afterInteractive"
        />
        {/* Load Three.js first, then Vanta effects in sequence */}
        <Script
          id="vanta-loader"
          strategy="afterInteractive"
          dangerouslySetInnerHTML={{
            __html: `
(function() {
  var s = document.createElement('script');
  s.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js';
  s.onload = function() {
    var f = document.createElement('script');
    f.src = 'https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.fog.min.js';
    document.head.appendChild(f);
    var c = document.createElement('script');
    c.src = 'https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.clouds.min.js';
    document.head.appendChild(c);
  };
  document.head.appendChild(s);
})();
            `,
          }}
        />
      </body>
    </html>
  )
}
