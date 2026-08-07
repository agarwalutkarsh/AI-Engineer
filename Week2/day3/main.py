import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"API Key: {GROQ_API_KEY}")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

client = Groq(api_key=GROQ_API_KEY)
model="llama-3.3-70b-versatile"

RESUME = """
UTKARSH AGARWAL 
Frontend Developer 
CAREER SUMMARY 
Results-driven React Frontend Developer with 4 years of experience in designing and developing robust, 
user-friendly web applications using React.js and Next.js. Experienced in integrating APIs, LLM-powered 
features, using AI coding tools, building scalable component architectures with a track record of delivering 
high-quality code aligned with best practices. Looking for a challenging role to t to build scalable and efficient 
user experiences and to explore innovative technologies in a dynamic team environment 
EXPERTISE 
Responsive Web Development  •  Performance Optimization & Code Splitting  •  Scalable Frontend 
Architecture  •  LLM API Integration & AI-powered Features  •  State Management & Data Flow  •  UI 
Implementation from Figma / Design Spec  •  Version Control & Collaboration 
TECHNICAL SKILLS 
React.js  •  Next.js •  Javascript  •  Redux  • Context API  •  Material UI  •  Tailwind CSS  •  Git/ Github •  Agile/ 
Scrum •  API Integration •  Node Js •  Express JS •  Azure Devops 
PROFESSIONAL EXPERIENCE 
Software Engineer – Frontend Developer | Planetcast Media Services Limited | Noida, India  Feb 2025 – 
Present 
• Key role in developing and maintaining Contido, a cloud-native Media Asset Management platform 
streamlining media supply chain and production workflows. 
• Contributed in integrating Gemini 2.5 Pro to build an AI-powered semantic tag generation feature for 
video assets, reducing the manual effort of adding context-aware tags. 
• Responsible for developing production-ready features using React.js and Next.js with a scalable, reusable 
component architecture. 
• Leveraged GitHub Copilot to decrease development time, improving code throughput and reducing 
boilerplate across the codebase. 
• Implemented workflow-driven interfaces using React Flow, allowing users to design and configure 
complex product workflows with custom interactive nodes. 
• Migrated the frontend build system to Webpack 5 and implemented Microfrontend Architecture using 
Module Federation, achieving 20% faster builds and independent deployments for 2 modules. 
• Participated in code reviews to ensure code quality, logic optimization, and best practices, resulting in a 
25% decrease in code errors. 
Frontend Developer (Cloud Engineer II) | Insight | Gurugram, India  Jul 2022 – Feb 2025 
• Built and maintained frontend applications, improving performance using modern web practices and 
caching strategies. 
• Created complex, interactive data visualizations using Chart.js for user-focused analytics interfaces. 
• Maintained high code quality through authentication implementation, code reviews, and logic optimization. 
• Participated in all Agile development processes and utilized Azure DevOps to ensure smooth feature 
delivery. 
EDUCATION 
B.Tech in Information Technology  |  KIIT Deemed University, Bhubaneswar, Odisha2018 – 2022 
CERTIFICATIONS 
Microsoft Azure Developer Associate (AZ-204)   •   AWS Certified Cloud Practitioner (CLF-C02) 
"""

JD = """About the job
About the Role



The Frontend Software Developer at EaseMyTrip.com will be tasked with crafting and enhancing the visual and interactive elements of web applications. This role involves using Angular along with other web technologies like HTML, CSS, and JavaScript to create responsive designs and integrate APIs. The developer will ensure cross-browser compatibility, optimize application performance, and maintain code quality through rigorous testing and documentation. This position requires staying updated with the latest in web development trends and technologies.


Role & responsibilities



Angular Development: Write clean and efficient code for web applications using Angular frameworks along with HTML, CSS, and JavaScript.
UI/UX Implementation: Convert design wireframes into interactive, functional web interfaces ensuring a seamless user experience.
Cross-Browser Compatibility: Guarantee consistent behavior and appearance of applications across various browsers.
API Integration: Work with backend APIs to fetch and display data effectively.
Performance Optimization: Enhance web application performance focusing on reducing load times and refining code.
Responsive Design: Develop responsive web designs to ensure optimal viewing across multiple devices.
Version Control: Utilize Git for efficient code management and team collaboration.
Documentation: Create detailed documentation related to code and development workflows.
Problem Solving: Identify and troubleshoot front end issues and bugs, proposing technical solutions when necessary.
Continuous Learning: Stay abreast of the latest industry trends in web development, including updates in technologies like Angular.


Preferred candidate profile



Experience: Minimum of 3 years in Angular-based frontend development.
Technical Skills: Proficient in Angular, JavaScript, TypeScript, HTML5, CSS3, Bootstrap, jQuery, and other modern JavaScript frameworks.
Agile Proficiency: Experienced with Agile and Scrum methodologies to enhance project delivery and efficiency.
Cross-Platform Coding: Skilled in writing code that works uniformly across different browsers and devices.
CI/CD Experience: Familiar with Continuous Integration and Continuous Delivery systems for streamlined development and deployment.
SEO Knowledge: Understands SEO principles and applies them to ensure enhanced application visibility.
Version Control and Databases: Proficient with Git and databases such as MySQL and MongoDB.
Independence and Proactivity: Capable of working independently and proactively to meet tight deadlines.
Communication Skills: Excellent communication and interpersonal skills to collaborate effectively with team members and stakeholders.
Innovative Approach: Able to introduce new technologies and practices to address business challenges and improve development processes
"""

def ask_llm (system_prompt, user_prompt) :
    messages = [
        {
            "role": "system",
            "content": system_prompt
         },
         {
             "role": "user",
             "content": user_prompt
         }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response

def extract_resume () :
    system_prompt = """
                    You are an expert HR assistant, and you have to extract only the skills from the resume of the candidate. Do not invent any new information or give answer from your side. Skills must be of that candidate.
                    """
    user_prompt = f"""
                    Extract the skills from this resume {RESUME}
                    """
    response = ask_llm(system_prompt, user_prompt)
    return response

candidate = extract_resume()
sleep(3)

def extract_jd () :
    system_prompt = """
                    You are an expert HR assistant, and you have to extract the skills from the jd. Do not invent any new information or give answer from your side. Skills must be of that JD.
                    """
    user_prompt = f"""
                    Extract the skills from this JD {JD}
                    """
    response = ask_llm(system_prompt, user_prompt)
    return response

jd= extract_jd()
sleep(3)

def match_skills () :
    system_prompt = f"""
                    You are an expert HR assistant, and you have the skills of both Candidate and JD. Your only job is to match the skills and score them from 1 to 100. do not invent any new informatation. Justify your answer whether this candidate is right for interview or not. {candidate} {jd}
                    """
    user_prompt = f"""
                    Match the skills of the candidate and jd.
                    """
    response = ask_llm(system_prompt, user_prompt)
    return response

candidate_match = match_skills()

# print(candidate.choices[0].message.content)
# print(jd.choices[0].message.content)
print(candidate_match.choices[0].message.content)