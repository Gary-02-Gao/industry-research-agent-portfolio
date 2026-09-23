// @vitest-environment jsdom
import { describe, expect, test } from 'vitest'
import { sanitizeMarkdownHtml } from './sanitize'
import { xssCorpus } from './xss-corpus'

describe('ThreatBench-50 markdown sanitization', () => {
  test('corpus contains exactly 50 independent payloads', () => {
    expect(xssCorpus).toHaveLength(50)
  })

  test.each(xssCorpus.map((payload, index) => [index + 1, payload] as const))(
    'payload %i cannot retain an executable sink', (_index, payload) => {
      const clean = sanitizeMarkdownHtml(payload)
      const root = document.createElement('div')
      root.innerHTML = clean
      expect(root.querySelector('script,style,svg,math,iframe,object,embed,form')).toBeNull()
      for (const element of root.querySelectorAll('*')) {
        for (const attribute of Array.from(element.attributes)) {
          expect(attribute.name.toLowerCase().startsWith('on')).toBe(false)
          expect(attribute.name.toLowerCase()).not.toBe('style')
          expect(attribute.name.toLowerCase()).not.toBe('srcset')
          if (attribute.name === 'href' || attribute.name === 'src' || attribute.name === 'action') {
            expect(attribute.value).not.toMatch(/^(?:javascript|vbscript|file|blob|data:text|data:image\/svg)/i)
          }
        }
      }
    },
  )
})
