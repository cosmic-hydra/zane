import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ZANE - AI Drug Discovery Platform',
  description: 'AI-native pharmaceutical operating system for autonomous molecular engineering',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-zane-dark text-white min-h-screen antialiased">
        <nav className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-3">
                <span className="text-2xl font-bold bg-gradient-to-r from-zane-primary to-zane-secondary bg-clip-text text-transparent">
                  ZANE
                </span>
                <span className="text-sm text-gray-400">AI Drug Discovery</span>
              </div>
              <div className="flex items-center space-x-6">
                <a href="/" className="text-gray-300 hover:text-white transition-colors text-sm">
                  Dashboard
                </a>
                <a href="/portal" className="text-gray-300 hover:text-white transition-colors text-sm">
                  Patient Portal
                </a>
              </div>
            </div>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
