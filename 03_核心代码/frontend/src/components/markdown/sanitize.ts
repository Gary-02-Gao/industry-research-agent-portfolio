import DOMPurify from 'dompurify'

export function sanitizeMarkdownHtml(html: string) {
  const sanitized = DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['script', 'style', 'svg', 'math', 'iframe', 'object', 'embed', 'form'],
    FORBID_ATTR: ['style', 'srcset'],
    ALLOW_DATA_ATTR: false,
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|[/?#.]|data:image\/(?:png|jpeg|gif|webp);base64,)/i,
  })

  // DOMPurify intentionally has a compatibility allowance for data: URLs on
  // image-like elements. Apply the product's narrower protocol policy after
  // sanitization so SVG/data:text cannot survive that special case.
  const template = document.createElement('template')
  template.innerHTML = sanitized
  const bitmapData = /^data:image\/(?:png|jpeg|gif|webp);base64,[a-z0-9+/=]+$/i
  const webOrRelative = /^(?:(?:https?):|[/?#.]|$)/i
  const linkTarget = /^(?:(?:https?|mailto):|[/?#.]|$)/i
  for (const element of template.content.querySelectorAll('*')) {
    for (const attribute of ['href', 'src', 'action']) {
      if (!element.hasAttribute(attribute)) continue
      const value = element.getAttribute(attribute)?.trim() ?? ''
      const allowed = attribute === 'href'
        ? linkTarget.test(value)
        : webOrRelative.test(value) || (element.tagName === 'IMG' && bitmapData.test(value))
      if (!allowed) element.removeAttribute(attribute)
    }
  }
  return template.innerHTML
}
