import { FileTextOutlined } from '@ant-design/icons'
import { useState } from 'react'
import styles from './search-results.module.scss'

interface SearchResult {
  id: string
  title: string
  source: string
  date?: string
  url?: string
  snippet?: string
}

interface SearchResultsProps {
  data?: SearchResult[]
}

export default function SearchResults({ data }: SearchResultsProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const openResult = (item: SearchResult) => {
    if (item.url?.startsWith('local://')) {
      setExpandedId(current => current === item.id ? null : item.id)
    } else if (item.url && /^https?:\/\//i.test(item.url)) {
      window.open(item.url, '_blank', 'noopener,noreferrer')
    }
  }
  if (!data?.length) {
    return (
      <div className={styles.empty}>
        <FileTextOutlined className={styles.emptyIcon} />
        <span>暂无搜索结果</span>
      </div>
    )
  }

  return (
    <div className={styles.list}>
      {data.map((item) => (
        <div
          key={item.id}
          className={styles.item}
          role="button"
          tabIndex={0}
          aria-label={`${item.title}：${item.url?.startsWith('local://') ? '展开或收起证据片段' : '打开来源'}`}
          aria-expanded={item.url?.startsWith('local://') ? expandedId === item.id : undefined}
          onClick={() => openResult(item)}
          onKeyDown={event => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              openResult(item)
            }
          }}
        >
          <div className={styles.icon}>
            <FileTextOutlined />
          </div>
          <div className={styles.content}>
            <div className={styles.title}>{item.title}</div>
            <div className={styles.meta}>
              <span className={styles.source}>{item.source}</span>
              {item.date && <span className={styles.date}>{item.date}</span>}
            </div>
            {item.snippet && <div className={`${styles.snippet} ${expandedId === item.id ? styles.expanded : ''}`}>{item.snippet}</div>}
          </div>
          <div className={styles.arrow}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </div>
      ))}
    </div>
  )
}
