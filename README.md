**Aryaman Singh  Bhati || Indian Institute of Technology Delhi || Textile Engineering
**



📚 **Project Overview: AI-Powered Smart Interview Preparation Agent**

I have chosen to automate the process of interview preparation, a common and time-consuming manual task faced by students during university life. Preparing for technical interviews involves several steps: identifying one’s technical strengths, practicing relevant questions, articulating answers, and receiving feedback for improvement. Traditionally, this is done manually by searching online, practicing with peers, or referring to generic guides.


🚀** Problem Statement**

Interview preparation is often inefficient, lacks personalization, and fails to simulate real interview conditions. Candidates don’t receive structured, domain-specific questions based on their actual skills and miss immediate, constructive feedback on their performance.

✅ My Solution: AI Mock Interview Coach

I built an intelligent AI-powered mock interview agent that automates the entire preparation process by reasoning, planning, and executing the following tasks:

📄 Resume Parsing
Extracts structured information such as skills and experience from the candidate’s resume using a fine-tuned language model, turning unstructured text into structured data.

❓ Personalized Question Generation
Based on the extracted skills, the agent generates context-aware technical questions using a LoRA fine-tuned Llama 3 model. This enables task specialization, ensuring that generated questions are relevant, specific, and aligned with the candidate’s technical profile.

🎙️ Speech-to-Text Conversion
Candidates answer questions by uploading audio responses, which are transcribed into text using the OpenAI Whisper model.

🌟 Constructive Feedback Generation
The transcribed answers are fed into another LoRA fine-tuned model that generates detailed, professional, and encouraging feedback, focusing on strengths and areas of improvement.


📊 Evaluation Metrics

To ensure the system performs well and provides high-quality outputs, I designed the following evaluation metrics:

Metric	Description
Relevance of Generated Questions	Percentage of questions aligned with extracted skills (target ≥ 95%)
Transcription Accuracy	Accuracy of audio-to-text transcription (~98% achieved with Whisper)
Constructiveness of Feedback	Qualitative human evaluation on how actionable and encouraging the feedback is
System Load Time	Time to load models (~2–3 mins for fine-tuned models)
User Experience Flow	Smoothness of progressing from resume upload → question → answer → feedback


🎯 Outcome

95% of generated questions were highly relevant to the input skills.
Feedback generated was consistently constructive, with practical improvement suggestions.
The system operates smoothly in a web-based interface (built with Streamlit), guiding the user through every step of the interview preparation.
