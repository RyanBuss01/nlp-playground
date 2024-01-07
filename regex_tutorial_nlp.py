import re

chat1 = "you're so annoying, 1234567891, abc@xyz.com"
chat2 = "here it is: (123)-456-7891, abc@xyz.com"
chat3 = "yes, phone: 1234567891 email: abc@xyz.com"

pattern = '\d{10}|\(\d{3}\)-\d{3}-\d{4}'

matches = re.findall(pattern, chat3)
print(matches)