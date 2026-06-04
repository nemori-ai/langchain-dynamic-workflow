import type { Metadata } from "next";
import "./globals.css";
import { Inter } from "next/font/google";
import React from "react";
import { NuqsAdapter } from "nuqs/adapters/next/app";

const inter = Inter({
  subsets: ["latin"],
  preload: true,
  display: "swap",
});

export const metadata: Metadata = {
  title: "Agent Chat",
  description: "Agent Chat UX by LangChain",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // `suppressHydrationWarning` on <html> only: some browser extensions inject
    // attributes (e.g. `data-redeviation-bs-uid`) onto the document element before
    // React hydrates, which the server HTML cannot match. This suppresses that
    // one-level attribute mismatch without masking real hydration bugs in children.
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <NuqsAdapter>{children}</NuqsAdapter>
      </body>
    </html>
  );
}
