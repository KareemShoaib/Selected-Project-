import html

with open('csv/raw/test.csv', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('&amp;', '&')
content = content.replace('&#39;', "'")
content = content.replace('&quot;', '""')
content = content.replace('&lt;', '<')
content = content.replace('&gt;', '>')

content = html.unescape(content)
with open('csv/clean/test.csv', 'w', encoding='utf-8') as f:
    f.write(content)

print('All HTML entities have been decoded!') 