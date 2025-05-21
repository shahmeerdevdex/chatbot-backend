const {
  maxNameLength,
  nameRegex,
  maxEmailLength,
  emailRegex,
  maxPasswordLength,
  strongPasswordRegex,
  maxPhoneNoLength,
  phoneNoRegex,
  agentNameRegex,
  notNullRegex,
  reqQuesRegex,
  areaCodeRegex,
  didTypeRegex,
  jobTypeRegex,
  numberRegex,
  datetimeRegex,
  callLimitRegex,
  limitRegex,
  countryRegex,
  intPhoneRegex,
  callNumberPattern,
  subTypeRegex,
  tempTypeRegex,
} = require("./regex");

const validationRules = {
  name: {
    maxLength: maxNameLength,
    regex: nameRegex,
    errorMessage: "Invalid name.",
  },
  email: {
    maxLength: maxEmailLength,
    regex: emailRegex,
    errorMessage: "Invalid email address.",
  },
  password: {
    maxLength: maxPasswordLength,
    regex: strongPasswordRegex,
    errorMessage: "Invalid password.",
  },
  currentPass: {
    maxLength: maxPasswordLength,
    regex: strongPasswordRegex,
    errorMessage: "Invalid current password.",
  },
  newPass: {
    maxLength: maxPasswordLength,
    regex: strongPasswordRegex,
    errorMessage: "Invalid new password.",
  },
  phoneNo: {
    maxLength: maxPhoneNoLength,
    regex: phoneNoRegex,
    errorMessage: "Invalid Phone Number.",
  },
  agentName: {
    maxLength: maxNameLength,
    regex: agentNameRegex,
    errorMessage: "Invalid Employee Name.",
  },
  templateId: {
    maxLength: 1000000,
    regex: notNullRegex,
    errorMessage: "Invalid Industry Template",
  },
  agentType: {
    maxLength: maxNameLength,
    regex: notNullRegex,
    errorMessage: "Invalid Employee Type.",
  },
  language: {
    maxLength: maxNameLength,
    regex: notNullRegex,
    errorMessage: "Invalid Language.",
  },
  voiceType: {
    maxLength: maxNameLength,
    regex: notNullRegex,
    errorMessage: "Invalid Voice Type.",
  },
  templateName: {
    maxLength: maxNameLength,
    regex: agentNameRegex,
    errorMessage: "Invalid Template Name.",
  },
  companyQues: {
    maxLength: 1000,
    regex: reqQuesRegex,
    errorMessage: "Invalid Company Question.",
  },
  greetQues: {
    maxLength: 1000,
    regex: reqQuesRegex,
    errorMessage: "Invalid Greeting Question.",
  },
  rulesQues: {
    maxLength: 1000,
    regex: reqQuesRegex,
    errorMessage: "Invalid Rules Question.",
  },
  agendaQues: {
    maxLength: 1000,
    regex: reqQuesRegex,
    errorMessage: "Invalid Agenda Question.",
  },
  eligibilityQues: {
    maxLength: 1000,
    regex: reqQuesRegex,
    errorMessage: "Invalid Eligibility Question.",
  },
  templateDetails: {
    maxLength: 1000,
    regex: reqQuesRegex,
    errorMessage: "Invalid Template Details.",
  },
  qualifyingQues: {
    maxLength: 1000,
    regex: reqQuesRegex,
    errorMessage: "Invalid Qualifying Question.",
  },
  areaCode: {
    maxLength: 1000000,
    regex: areaCodeRegex,
    errorMessage: "Area code must contain only digits.",
  },
  limit: {
    maxLength: 1000000,
    regex: limitRegex,
    errorMessage: "Limit must be greater than 0.",
  },
  DIDType: {
    maxLength: 10,
    regex: didTypeRegex,
    errorMessage: "Please select a valid DID Type.",
  },
  country: {
    maxLength: 5,
    regex: countryRegex,
    errorMessage: "Invalid Country.",
  },
  phoneNumber: {
    maxLength: 20,
    regex: intPhoneRegex,
    errorMessage: "Invalid Phone Number.",
  },
  listName: {
    maxLength: maxNameLength,
    regex: nameRegex,
    errorMessage: "Invalid List Name.",
  },
  callNumberPattern: {
    maxLength: 20,
    regex: callNumberPattern,
    errorMessage:
      "Invalid Phone Number. Only numbers and at least 8 digits long.",
  },
  jobType: {
    maxLength: 15,
    regex: jobTypeRegex,
    errorMessage: "Invalid Job Type.",
  },
  callLimit: {
    maxLength: 15,
    regex: callLimitRegex,
    errorMessage: "Invalid Call Limit.",
  },
  startDateTime: {
    maxLength: 30,
    regex: datetimeRegex,
    errorMessage: "Invalid Start Date Time.",
  },
  endDateTime: {
    maxLength: 30,
    regex: datetimeRegex,
    errorMessage: "Invalid End Date Time.",
  },
  subType: {
    maxLength: 10,
    regex: subTypeRegex,
    errorMessage: "Invalid subscription type.",
  },
  templateType: {
    maxLength: 15,
    regex: tempTypeRegex,
    errorMessage: "Invalid template type.",
  },
};

const validateInput = (value, type) => {
  const { maxLength, regex, errorMessage } = validationRules[type] || {};
  if (type === "phoneNo" && !value) return null;
  if (type === "areaCode" && !value) return null;
  if (type === "limit" && !value) return null;
  if (Array.isArray(value)) {
    for (let item of value) {
      if (
        !item ||
        item.trim().length === 0 ||
        item.trim().length > maxLength ||
        !regex.test(item.trim())
      ) {
        return errorMessage;
      }
    }
  } else {
    if (
      !value ||
      value.trim().length === 0 ||
      value.trim().length > maxLength ||
      !regex.test(value.trim())
    ) {
      return errorMessage;
    }
  }
  return null;
};

module.exports = validateInput;
